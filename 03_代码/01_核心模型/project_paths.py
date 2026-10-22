"""项目路径：代码与有效数据留在本目录，结果和资料放在外部分类目录。"""
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORAGE_ROOT = PROJECT_ROOT.parent
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = STORAGE_ROOT / '项目结果'
FIGURES_DIR = RESULTS_DIR / 'figures'
PAPER_DIR = RESULTS_DIR / 'paper_outputs'
TABLES_DIR = RESULTS_DIR / 'tables'
DOCS_DIR = STORAGE_ROOT / '项目文档'
SOURCES_DIR = STORAGE_ROOT / '数据来源'
BACKUP_DIR = STORAGE_ROOT / '历史存档' / 'pass_data'
LOGS_DIR = STORAGE_ROOT / '运行记录'
