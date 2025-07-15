
#file_listにはファイル名のリストが入っています。
#それらを使って、AIにファイルを解析してもらうプロンプトを作成してください。
#AIの出力はプロジェクトの概要（description）と工夫されている点(improvements)、
#それらをまとめたマークダウン形式のドキュメント(markdown)の3つにしてほしいです。

#出力が安定しない場合は、プロジェクトの概要と工夫されている点について生成させます。
#その後それらを入力として、マークダウン形式のドキュメントを生成させてください。

#add_promptがNone出ない場合は、プロンプトにtextを追加してください。

def process(file_list,add_prompt=None):
    
    output = {
        'description': 'get_api_output:test',
        'improvements': 'get_api_output:test',
        'markdown': '# get_api_output:test'
    }
    return output