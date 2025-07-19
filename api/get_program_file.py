# file_listはzipファイルを解凍して得られたファイルです。
# この中から、プログラムのファイルだけを抽出して、ファイル名のリストを返します。
# プログラムのファイルになりそうな拡張子をprog_extsに定義します。
# この拡張子を持つファイルだけfile_listに追加します。


def process(file_list):
    file_list_t = []
    prog_exts = [
    # コンパイル系
    ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx",
    ".cs",
    ".java", ".class", ".jar",
    ".go",
    ".rs",
    ".swift", ".m", ".mm",
    ".kt", ".kts",
    ".scala",
    ".d",
    ".hs", ".lhs",
    ".f", ".for", ".f90", ".f95", ".f03", ".f08",
    ".pas", ".pp", ".dpr", ".dfm",
    ".adb", ".ads",
    ".ml", ".mli",
    ".erl", ".ex", ".exs",
    ".nim",
    ".zig",
    ".jl",

    # スクリプト系
    ".py", ".pyw", ".pyc", ".pyo",
    ".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx",
    ".php", ".phtml", ".php3", ".inc",
    ".pl", ".pm",
    ".rb", ".erb",
    ".lua",
    ".tcl", ".tk",
    ".sh", ".bash",
    ".ps1", ".psm1",
    ".groovy",
    ".coffee",
    ".dart",

    # マークアップ／ノートブック
    ".html", ".htm", ".xml",
    ".xsl", ".xslt",
    ".md", ".markdown",
    ".tex",
    ".ipynb",
    ".vue", ".svelte",

    # スタイルシート・設定・データ
    ".css", ".scss", ".sass", ".less",
    ".json", ".json5", ".yaml", ".yml", ".toml",
    ".graphql", ".gql",
    ".sql", ".pls", ".pks", ".pkb",
    ".ini",
    ]
    
    for i in file_list:
        if any(i.endswith(ext) for ext in prog_exts):
            file_list_t.append(i)

    file_list = file_list_t
    file_list.sort()
    
    return file_list
