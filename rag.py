import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
import chardet
import uuid


class RAGTool:
    def __init__(self, file_path):
        self.client = chromadb.PersistentClient(f"{file_path}/chroma_db")
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="G:/big_model/model/BAAI/bge-small-zh-v1.5"
        )

        self.collection = self.client.get_or_create_collection("docs", embedding_function=self.embedding_func)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def add(self, content: str):
        chunks = self.text_splitter.split_text(content)
        batch_size = 100  # 每批处理100个
        metadatas = [{"source": "chat", "chunk_index": i} for i in range(len(chunks))]
        for i in tqdm(range(0, len(chunks), batch_size), desc="向量化并存储"):
            batch_end = min(i + batch_size, len(chunks))
            self.collection.add(
                documents=chunks[i:batch_end],
                metadatas=metadatas[i:batch_end],
                ids=[str(uuid.uuid4()) for i in range(len(chunks))]
            )
        print(f"✓ 完成记忆")

    def load_document(self, file_path: str, force_reload=False):
        """加载文档并分片存储
        
        Args:
            file_path: 文件路径
            force_reload: 是否强制重新加载(默认False,如果已存在则跳过)
        """
        # 检查是否已经加载过
        if not force_reload:
            existing_data = self.collection.get()
            if len(existing_data['ids']) > 0:
                print(f"Collection 中已有 {len(existing_data['ids'])} 个文档,跳过加载")
                return

        # 自动检测文件编码
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            confidence = result['confidence']
            print(f"检测到文件编码: {encoding} (置信度: {confidence:.2%})")

        # 使用检测到的编码读取文件
        content = raw_data.decode(encoding)

        # 使用 text_splitter 分片
        chunks = self.text_splitter.split_text(content)
        print(f"文档已分片为 {len(chunks)} 个块")

        # 为每个分片生成唯一 ID
        ids = [f"doc_{i}" for i in range(len(chunks))]
        metadatas = [{"source": file_path, "chunk_index": i} for i in range(len(chunks))]

        # 批量添加所有分片,显示进度条
        batch_size = 100  # 每批处理100个
        for i in tqdm(range(0, len(chunks), batch_size), desc="向量化并存储"):
            batch_end = min(i + batch_size, len(chunks))
            self.collection.add(
                documents=chunks[i:batch_end],
                metadatas=metadatas[i:batch_end],
                ids=ids[i:batch_end]
            )

        print(f"✓ 文档处理完成,共存储 {len(chunks)} 个分片")

    def query(self, sentence):
        """
        从本地查询指定的数据

        Args:
            sentence: 问题

        Returns:
            最相关的三个数据
        """
        result = self.collection.query(query_texts=[sentence], n_results=1)
        return result
