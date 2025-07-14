
def process(prj_id,text):
    """
    追加の生成処理を行う
    """
    #prj_idを使って、必要なデータをデータベースから参照してプログラムファイルを抽出してください。
    #プログラムファイルを抽出できたら、AIのAPIを呼び出して生成させてください。
    #textも使って生成してください。
    #AIには以下の項目について生成させてください。
    #・プログラムの概要
    #・工夫している内容
    #・markdown形式での出力
    #出力の内容は辞書型でまとめてください。
    #例: output = {'description': '概要の内容', 'improvements': '工夫の内容', 'markdown': '生成されたMarkdown内容'}
    output = {
        'description': 'generate_add:test',
        'improvements': 'generate_add:test',
        'markdown': '# generate_add:test'}
    return output