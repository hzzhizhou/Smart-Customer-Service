"""
企业级全局配置：集中管理所有参数，避免硬编码，便于环境切换
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 项目路径配置
BASE_DIR = Path(__file__).parent.parent

# 加载环境变量（企业级密钥管理）
# 显式绑定到 .env 所在目录，与进程启动时的工作目录无关（避免 cwd 不同导致找不到密钥）
load_dotenv(BASE_DIR / ".env")
VECTOR_DB_DIR = BASE_DIR / "vector_db-sql"   # 向量库存储目录
STOP_WORDS = BASE_DIR / "config"/"stop_words.txt"

# LLM/嵌入模型配置（企业级版本锁定）
LLM_MODEL = "deepseek-v4.1-flash"#换更快的模型：qwen-turbo 比 qwen3-max 快很多，效果差异不大。
LLM_TEMPERATURE = 0.01  # 无幻觉：固定温度
LLM_SEED = 42        # 固定随机种子（结果稳定）
# 单次 LLM 调用超时（秒）。流式下该值即「两个 chunk 之间的读超时」：
# 模型挂住不出字时抛异常，由生成层/Agent 的兜底话术接住，避免一轮请求无限期占住连接。
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))

# ====================== 嵌入模型配置 ======================
EMBEDDING_BACKEND = "dashscope"    # "dashscope" 或 "local"（云端约0.5s，本地CPU bge-base 20s+）
# LOCAL_EMBEDDING_MODEL = "D:/RAG-Windows/AI大模型与智能体开发/models/bge-small-zh-v1.5"#---512维
LOCAL_EMBEDDING_MODEL = "D:/RAG-Windows/AI大模型与智能体开发/models/bge-base-zh-v1.5"#--768维,本地下载的模型地址
# 云端模型（仅当 EMBEDDING_BACKEND="dashscope" 时使用）
DASHSCOPE_EMBEDDING_MODEL = "qwen3.7-text-embedding-flash"# 1024维
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")  # 从.env读取

# ====================== LLM OpenAI 兼容模式 ======================
# 统一改用 DashScope 的 OpenAI 兼容端点（compatible-mode）。key、模型均为通义账号，仅调用协议不同。
# LLM 工厂函数见 utils/llm_factory.py（config 仅保留纯配置，不含运行时逻辑）
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# 数据层配置
CHUNK_SIZE = 768               # 分块大小（适配LLM上下文）
CHUNK_OVERLAP = 128             # 块重叠（避免上下文丢失）
MIN_CHUNK_SIZE = 50             # 最小块长度（低于该值的块不入库，避免无效 embedding）
MAX_FILE_SIZE = 10 * 1024 * 1024  # 最大文件大小（10MB，规避内存溢出）
# 单请求体上限（含 multipart 上传）：超限直接 413，不读 body、不进业务路由。
# 必须大于 MAX_FILE_SIZE，给 multipart 边界与表单头留余量；
# 上传文件本身在路由里另按 MAX_FILE_SIZE 卡一遍（防止单文件超限靠分块头部撑大）。
MAX_REQUEST_BODY_SIZE = int(os.getenv("MAX_REQUEST_BODY_SIZE", str(12 * 1024 * 1024)))
ALLOWED_EXTENSIONS = [".txt", ".pdf", ".docx", ".xlsx", ".md", ".csv"]  # 允许的文件格式

# 建单幂等窗口（秒）：同账号 + 同订单 + 同分类在此窗口内视为「同一次申请」，
# 由 tickets.idempotency_key 唯一键兜底，并发/重试只会留下第一张工单。
# 窗口外的再次提交会新建工单，避免一次误操作把该业务永久锁死。
TICKET_IDEMPOTENCY_WINDOW = int(os.getenv("TICKET_IDEMPOTENCY_WINDOW", "300"))

# metadata 预过滤（按问题关键词推断文档类别）的全局策略。
# 该开关决定召回范围，必须只有一个来源：/rag/stream、Agent 的 search_knowledge、
# 知识兜底与离线评测全部走这里，否则同一问题在不同入口召回范围不同，评测结论不可比。
AUTO_FILTER_DEFAULT = os.getenv("AUTO_FILTER", "true").lower() != "false"


# ====================== 分块策略配置 ======================
CHUNKING_STRATEGY = "recursive"  # 可选: recursive, semantic, sliding_window, parent_child, combined_splitter
SEMANTIC_EMBEDDING_MODEL = "D:/RAG-Windows/AI大模型与智能体开发/models/bge-base-zh-v1.5"#--768维
SEMANTIC_CHUNK_THRESHOLD = 0.75   # 句子相似度阈值（低于则切分）
SEMANTIC_BUFFER_SIZE = 2           # 切分时前后保留的句子数

# 滑动窗口分块参数
SLIDING_WINDOW_STEP = 256          # 滑动步长（若未设置，则使用 CHUNK_SIZE - CHUNK_OVERLAP）

# 父子分块参数
PARENT_CHUNK_SIZE = 2000           # 父块大小
PARENT_CHUNK_OVERLAP = 200         # 父块重叠
CHILD_CHUNK_SIZE = 400             # 子块大小
CHILD_CHUNK_OVERLAP = 40           # 子块重叠

TABLE_CHUNK_SIZE = 1000            # 表格块大小（表头 + 约 5~6 行）

# 嵌入模型单条输入上限（token）：分块长度必须 ≤ 该值，否则末尾内容会被嵌入模型静默截断，
# 召回随之下降却没有任何报错 —— 属于「配置一换就坏」的隐患（PROJECT_AUDIT §4.2(3)）。
# 按后端取上限：dashscope 的 qwen text-embedding 上限 8192；
# 本地 bge 系列 512（中文约 1 字 1 token）。
EMBEDDING_MODEL_MAX_TOKENS = 8192 if EMBEDDING_BACKEND == "dashscope" else 512
# 分块后追加的【标题 > 章节】前缀预留长度（见 ingestion/splitter/base.prepend_section_path）
CHUNK_PREFIX_OVERHEAD = 60


def validate_chunk_config() -> list:
    """启动校验：分块配置会不会被嵌入模型静默截断（PROJECT_AUDIT §4.2(3)）。

    返回告警文案列表（空列表 = 自洽）。只告警、不 raise：切回本地 bge 是使用者的
    显式选择，不该让服务起不来；但必须在启动时把「会被截断」这件事喊出来，
    否则问题只体现在检索质量上、看板无信号。

    注意：这里只做校验，**不重建索引** —— 分块参数变更须重新入库才会生效。
    """
    warnings = []
    for name, size in (("CHUNK_SIZE", CHUNK_SIZE), ("TABLE_CHUNK_SIZE", TABLE_CHUNK_SIZE)):
        budget = size + CHUNK_PREFIX_OVERHEAD   # 标题前缀也计入块长度
        if budget > EMBEDDING_MODEL_MAX_TOKENS:
            warnings.append(
                f"{name}={size}（含标题前缀预留 {CHUNK_PREFIX_OVERHEAD}，合计 {budget}）"
                f"超过嵌入模型上限 {EMBEDDING_MODEL_MAX_TOKENS} token："
                f"末尾内容会被静默截断，请调小该值或改用上限更大的嵌入模型"
            )
    return warnings


# 检索层配置
RETRIEVER_K = 10                # 检索返回数量

# ============ 融合排序（多路召回 → 一个排序） 仅混合检索需要=========
# rrf：标准 RRF，两路按名次累加 1/(k+rank)，与分数尺度无关、不需要权重（默认）
# weighted：两路分数各自 Min-Max 归一化后按 BM25_WEIGHT/VECTOR_WEIGHT 加权
# 两者同属融合层、二选一（不串联）；融合之后才做重排（Cross-Encoder）。
FUSION_METHOD = "rrf"
RRF_K = 60                      # RRF 平滑常数：越大越弱化相邻名次间的得分差距
BM25_WEIGHT = 0.4              # weighted 策略下的 BM25 权重
BM25_SCORE_THRESHOLD = 0.1      #阙值过滤

# ==========文档类别预过滤（metadata filter） ==============
# 检索前按问题关键词推断文档类别，先在 metadata 上缩小范围再检索（类别字段见
# ingestion.loader.mysql_data_loader.derive_doc_category，落在每个块的 doc_category 上）；无关键词命中则不过滤，
# 过滤后无结果时检索层会自动回退全库，避免"把正确文档挡在门外"。
DOC_CATEGORY_PRIORITY = ["phone", "product", "faq"]   # 命中多个类别时按此优先级取一个
DOC_CATEGORY_KEYWORDS = {
    "phone":   ["手机", "iphone", "苹果", "华为", "小米", "mate", "机型", "屏幕", "续航", "影像"],
    "product": ["多少钱", "售价", "参数", "商品", "配件", "充电器", "数据线", "保护壳"],
    "faq":     ["退货", "退款", "运费", "发票", "保修", "客诉", "缺件", "价格保护", "客服", "订单状态"],
}

VECTOR_WEIGHT = 0.6            # weighted 策略下的向量权重
ROUTE_MODE = "rule"            # 路由模式：rule/llm/hybrid

# ========== 重排（融合结果 → Cross-Encoder 精排） ==========
RERANK_TOP_N = 5
# cross_encoder：用本地 Cross-Encoder 重排；none：关闭重排，直接返回融合序（可灰度/排障）
RERANK_MODE = "cross_encoder"
# 本地重排模型目录（BAAI/bge-reranker-base 的完整权重，非 sentence-transformers 缓存）
RERANK_MODEL_PATH = os.getenv("RERANK_MODEL_PATH",
                              str(BASE_DIR.parent / "models" / "bge-reranker-base"))
# 送进 Cross-Encoder 的候选上限：CPU 下打分耗时与候选数成正比，截断以控延迟
# （RERANK_TOP_N 是最终返回数，本值是打分规模，应 ≥ RERANK_TOP_N）
RERANK_CANDIDATES = 20

# 置信度门控阈值（量纲 = 余弦相似度 cos）
CONFIDENCE_HIGH_THRESHOLD = 0.80    # top cos >= 该值且 margin 充足 → high，直接回答
CONFIDENCE_LOW_THRESHOLD = 0.55     # < 该值 → low，转人工
CONFIDENCE_MARGIN_THRESHOLD = 0.08  # top 与第二名差距 < 该值 → 排序不稳，high 降级为 medium
# margin 阈值随量纲同步放大（cos 尺度的同一条线约为旧 1/(1+d) 尺度的 1.6 倍），
# 不改会让「同一条分界线」在新的高分区间里变宽松，高分也被误降级。


# 查询扩展（HyDE / 多查询）由调用方按请求决定：
# /rag/stream 的 use_hyde 入参、Agent 与评估层显式传 False，故不设全局开关。

# ========== 细粒度业务意图分类器 ==========
# 在调用 Agent 前识别业务子意图，让 Agent 优先使用对应工具
INTENT_MODE = "hybrid"        # rule=纯规则; llm=纯LLM; hybrid=规则优先+LLM兜底
INTENT_LLM_FALLBACK_CONFIDENCE = 0.5  # 规则识别置信度低于该值时才走 LLM 兜底

# ========== DST 对话状态跟踪 ==========
# 轻量接入：请求前加载跨轮会话状态（意图/槽位/阶段），响应后 LLM 提取槽位回写。
# 置 False 时整体行为与未接入前一致（可灰度）。
DST_ENABLED = True


# 业务意图 → 优先工具 映射（供 Agent 决策提示 + 路由参考）
INTENT_TOOL_MAP = {
    "consult":    "search_knowledge",   # 知识咨询
    "return":     "search_knowledge",   # 退货（先查规则，必要时建工单）
    "refund":     "query_order",        # 退款（查退款进度，必要时查政策）
    "logistics":  "query_order",        # 物流
    "order":      "query_order",        # 订单状态
    "complaint":  "create_ticket",      # 投诉 → 直接转人工
    "chat":       "none",               # 闲聊，不调工具
}



# 评估层配置
EVAL_RUNS = 1                  # 评估次数（取平均，结果稳定）
EVAL_THRESHOLDS = {            # 企业级指标阈值（低于告警）
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "context_precision": 0.90,
    "context_recall": 0.95
}

# 部署层配置
API_HOST = "0.0.0.0"
API_PORT = 8000
CACHE_MAXSIZE = 1000           # 缓存最大条数
CACHE_TTL = 300                # RAG 响应缓存有效期（秒），TTL 内命中秒回，过期自动失效
RESPONSE_TIMEOUT = 30          # API响应超时时间
RATE_LIMIT_PER_MINUTE = 60     # 每个 API key 每分钟最大请求数（限流，防滥用）
# API 交互文档（/docs、/redoc、/openapi.json）：默认关闭，避免把接口面与内部模型暴露到公网；
# 本地演示可设 ENABLE_API_DOCS=true 打开。
ENABLE_API_DOCS = os.getenv("ENABLE_API_DOCS", "false").strip().lower() in ("1", "true", "yes", "on")
# 登录令牌有效期（小时）：过期的令牌查库即失效，返回 401 要求重新登录；默认 7 天。
TOKEN_TTL_HOURS = int(os.getenv("TOKEN_TTL_HOURS", "168"))
# 订单/工单/用户/会话等业务数据落在 MySQL（见 infrastructure/mysql_store.py），
# 连接参数取文件末尾的 MYSQL_* 配置。


# 外部依赖熔断器配置（保护系统不被下游故障拖垮）
CIRCUIT_FAILURE_THRESHOLD = 5   # 连续失败该次数后熔断（CLOSED→OPEN）
CIRCUIT_RECOVERY_TIMEOUT = 60  # 熔断后等待该秒数后进入半开探测（OPEN→HALF_OPEN）
CIRCUIT_HALF_OPEN_MAX_CALLS = 1  # 半开态允许的探测请求数（探测成功→CLOSED，失败→OPEN）

# 日志配置统一在 logs/log_config.py（级别/轮转/输出目标），此处不再重复声明。
# 该模块自带降级逻辑、可独立于 config 使用，故不反向依赖本文件。
# 文件侧 JSON 结构化输出（每行一个 JSON 对象，含 request_id）；
# 控制台始终为纯文本（含 request_id），便于开发阅读。False 可回退旧纯文本格式。
LOG_FORMAT_JSON = True

# Redis配置
REDIS_CONFIG = {
    # 原有配置（对话历史）。host/port 支持环境变量覆盖：
    # Docker 编排里应用容器通过服务名 redis 连接（见 docker-compose.yml 的 REDIS_HOST=redis），
    # 本地开发不设则保持 localhost 不变。
    "host": os.getenv("REDIS_HOST", "localhost"),
    "port": int(os.getenv("REDIS_PORT", "6379")),
    "password": os.getenv("Redis_password"),
    "db": 0,
    "key_prefix": "rag_chat_history:",     # 对话历史前缀
    "expire_days": 7,
    "socket_timeout": 10,
    "max_connections": 30,
}
#对话记忆层配置
SESSION_ID ="user_001"
CHAT_HISTORY_WINDOW = 6          # 多轮对话历史窗口（统一 Agent/RAG 两端，之前 RAG 取2、Agent 取6 不一致）
MAX_CONTEXT_TOKENS = 3500        # 上下文最大 token 数（LLM 计费/截断以 token 为准，之前用字符数 4000 不准）

# 输出合规护栏配置（客服场景必备：防止 LLM 做出未经授权的承诺）
OUTPUT_GUARD_ENABLED = True      # 总开关
GUARD_SENSITIVE_WORDS = [        # 敏感词：出现即拦截替换
    "辱骂", "诈骗", "转账", "银行卡号", "验证码",
]
GUARD_COMMITMENT_PATTERNS = [    # 需人工授权的承诺话术（正则）
    r"全额退款", r"免费换新", r"双倍赔偿", r"十倍赔偿",
    r"保证\s*\d+\s*天.*(?:到|送达)",   # 保证X天到货
    r"\d+倍.*赔偿",
]
GUARD_AMOUNT_PATTERN = r"¥\s*\d+(?:\.\d+)?"   # 金额提取正则（校验是否有依据）


# ---------------------------异步检索--------------------------------
# CPU核心数（自动适配）
import multiprocessing
CPU_CORES = multiprocessing.cpu_count()

# BM25检索（CPU密集）
# BM25_THREAD_POOL_SIZE 设置为 CPU 核心数的两倍，
# 适合 CPU 密集型检索（BM25 算法多线程并发），
# 通常这样可以充分利用多核资源，但避免线程过多导致频繁上下文切换。
BM25_THREAD_POOL_SIZE = CPU_CORES 
BM25_THREAD_POOL_NAME = "bm25-retriever-pool"
BM25_ASYNC_TIMEOUT = 10

# 向量检索（IO密集）
VECTOR_THREAD_POOL_SIZE = CPU_CORES * 2
VECTOR_THREAD_POOL_NAME = "vector-retriever-pool"
VECTOR_ASYNC_TIMEOUT = 10

HYBRID_ASYNC_TIMEOUT = 10


#-----------MySQL配置---------------------
# MySQL 配置
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))
MYSQL_USER = os.getenv("MYSQL_USER", "rag_user")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "123456")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "rag_db")