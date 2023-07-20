import os
import time
from threading import Thread
import warnings
warnings.filterwarnings('ignore')

from langchain import PromptTemplate
from langchain.llms import OpenAI
import tiktoken
from tiktoken.core import Encoding

import streamlit as st


# --- ストップウォッチ用のクラス---
class Stopwatch:
    def __init__(self):
        self._start_time = None
        self._elapsed_time = 0

    def start(self):
        self._start_time = time.time()

    def stop(self):
        if self._start_time is not None:
            self._elapsed_time += time.time() - self._start_time
            self._start_time = None

    def get_time(self):
        if self._start_time is not None:
            return self._elapsed_time + (time.time() - self._start_time)
        return self._elapsed_time
    
    
# ---アウトプットを表示するためのクラス---
class ProcessOutput:
    def __init__(self):
        self._value = None

    def set_value(self, value):
        self._value = value

    def get_value(self):
        return self._value

    def is_set(self):
        return self._value is not None


# ---ペルソナ作成用の関数---
# モデルを用意する
def read_model(model_name='gpt-3.5-turbo', max_tokens=4000, temperature=1):
    return OpenAI(model_name=model_name, 
                      temperature=temperature, 
                      max_tokens=max_tokens,
                      request_timeout=600,
                     )


# 日本語で出力するプロンプト
def make_prompt_persona_ja(product_name, product_price, product_features):
    prompt_template = """あなたは顧客心理に訴えかけ、問題解決に焦点を当てた効果的なセールスレターを書くプロフェッショナルのセールスライターです。日本語でペルソナを作成してください。
    
    # Step1. 商品やサービスの詳細を把握する
    - 商品やサービスの名称: {product_name}
    - 商品やサービスの価格: {product_price}
    - 商品やサービスの特徴や利点: {product_features}

    # Step2. 商品やサービスに関連する
    各認識レベルごとにペルソナを作成する  
    - 認識レベル5:
    問題を解決する商品の購入を検討している  
    - 認識レベル4:
    問題を解決できる商品を (いくつか) 知っている  
    - 認識レベル3:
    問題の解決策は知っているが商品の存在は知らない  
    - 認識レベル2:
    問題には気づいているが、 解決策の方法を知らない  
    - 認識レベル1:
    そもそも問題に気づいていない

    # ペルソナ作成時に考慮すべき要素:
    - 名前
    - 年齡
    - 性別
    - 職業
    - 年収
    - 地域
    - 家族構成
    - 趣味
    - 価値観
    - 課題や悩み
    
    これらの要素を組み合わせて、各認識レベルごとに具体的なペルソナを作成します。
    出力は箇条書きで、各要素ごとに改行してください。
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["product_name", "product_price", "product_features"])
    return prompt.format(product_name=product_name, product_price=product_price, product_features=product_features)


# トークン数を数える
def count_token(text, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    tokens_count = len(tokens)
    return tokens_count
    
    
# 翻訳しないで実行する
def process_ja(product_name, product_price, product_features, stopwatch, output):
    # プロンプト作成
    prompt_persona = make_prompt_persona_ja(product_name=product_name, product_price=product_price, product_features=product_features)
    tokens_ja = count_token(prompt_persona) # トークン数を数えておく

    # ペルソナを作成する
    llm_persona = read_model(max_tokens=4000-tokens_ja)  # 入力+出力<4096になるように設定
    output_persona = llm_persona(prompt_persona)
    
    # ストップウォッチを止める
    stopwatch.stop()

    # return output_persona_ja
    result = f"""
    
    {output_persona}
    """
    output.set_value(result)


# ペルソナリサーチで使う関数
## ペルソナを入れるプロンプト
def make_prompt_base(product_name, product_price, product_features, persona):
    prompt_template = """あなたは顧客心理に訴えかけ、問題解決に焦点を当てた効果的なセールスレターを書くプロフェッショナルのセールスライターです。
    
    # 対象となる商品やサービスの詳細
    - 商品やサービスの名称: {product_name}
    - 商品やサービスの価格: {product_price}
    - 商品やサービスの特徴や利点: {product_features}
    
    以下のペルソナについて深く分析してください。
    # ペルソナ:
    {persona}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["product_name", "product_price", "product_features", "persona"])
    return prompt.format(product_name=product_name, product_price=product_price, product_features=product_features, persona=persona)


## リサーチをするためのプロンプト
def make_prompt_research(base_prompt, research):
    prompt_template = """
    {base_prompt}
    {research}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["base_prompt", "research"])
    return prompt.format(base_prompt=base_prompt, research=research)


# リサーチを実行する
def process_research(product_name, product_price, product_features, persona, research, 
                     stopwatch, output):
    # プロンプト作成
    prompt_base = make_prompt_base(product_name=product_name, product_price=product_price, product_features=product_features, persona=persona)
    prompt_research = make_prompt_research(base_prompt=prompt_base, research=research)
    # トークン数を数えておく
    tokens_r = count_token(prompt_research)
    
    # ペルソナを分析する
    llm_analysis = read_model(max_tokens=4000-tokens_r)  # 入力+出力<4096になるように設定
    output_r = llm_analysis(prompt_research)
    
    # ストップウォッチを止める
    stopwatch.stop()
    
    # 出力
    result = f"""
    {output_r}
    """
    output.set_value(result)

    
# ---実装---

# セットアップ
st.set_page_config(layout="wide")
pagelist = ["ペルソナ作成", "ペルソナリサーチ", "出力結果"] #セレクトボックスのリストを作成
selector = st.sidebar.radio("Menu", pagelist) #サイドバーのセレクトボックスを配置
st.sidebar.markdown("""※注意  
ペルソナ作成 → ペルソナリサーチの順に実行してから、出力結果のページに飛ぶようにしてください""")

# 複数ページで使用する変数をセッションに追加する
if 'product_name' not in st.session_state:
    st.session_state['product_name'] = "低糖質のパン"
if 'product_price' not in st.session_state:
    st.session_state['product_price'] = "200円"
if 'product_features' not in st.session_state:
    st.session_state['product_features'] = "通常のパンよりも糖質量が少なく、ダイエットに効果的"
if 'persona' not in st.session_state:
    st.session_state['persona'] = 'default'
## 結果出力用にセッション保持する
if 'made_persona_prompt' not in st.session_state:
    st.session_state['made_persona_prompt'] = 'default'
if 'made_research_prompt' not in st.session_state:
    st.session_state['made_research_prompt'] = 'default'
if 'result_research' not in st.session_state:
    st.session_state['result_research'] = 'default'    
    

# ---ペルソナ作成---
if selector == pagelist[0]:
    # タイトル
    st.title("ペルソナを作成するよ！")


    # 変数を定義
    st.header('商品情報を入力してください')
    st.caption('デフォルトでテキストが入っていますが、適宜書き換えてください')
    st.session_state['product_name'] = st.text_input('商品やサービスの名称', value=st.session_state['product_name'])
    st.session_state['product_price'] = st.text_input('商品やサービスの価格', value=st.session_state['product_price'])
    st.session_state['product_features'] = st.text_input('商品やサービスの特徴や利点', value=st.session_state['product_features'])

    # 実行ボタン
    execute_make = st.button('ペルソナ作成')

    # ストップウォッチ用
    stopwatch = Stopwatch()
    timer_placeholder = st.empty()

    # アウトプット用
    output = ProcessOutput()

    # 文章作成
    if execute_make:
        st.info('作成には時間がかかることがあります。気長にお待ちください')
        # 時間計測開始
        stopwatch.start()
        
        # 変数を受け取っておく
        product_name = st.session_state['product_name']
        product_price = st.session_state['product_price']
        product_features = st.session_state['product_features']

        # プロンプトを表示する
        prompt_persona = make_prompt_persona_ja(product_name=product_name, 
                                                product_price=product_price, 
                                                product_features=product_features,)
        st.header('入力するプロンプト')
        st.code(prompt_persona)
        # 変数を保存
        st.session_state['made_persona_prompt'] = prompt_persona

        Thread(target=process_ja, args=(product_name, product_price, product_features, stopwatch, output)).start()
        st.header('作成されたペルソナ')

    while execute_make and not output.is_set():
        timer_placeholder.text('Timer: ' + str(round(stopwatch.get_time(), 2)))
        time.sleep(0.01)

    # 結果出力
    st.write(output.get_value())
    # 変数を保存
    st.session_state['persona'] = output.get_value()
    
    
# ---ペルソナリサーチ---
elif selector == pagelist[1]:
    # タイトル
    st.title("ペルソナを分析するよ！")


    # 変数を定義
    st.header('分析したいペルソナを入力してください')
    
    st.caption('デフォルトでテキストが入っていますが、適宜書き換えてください')
    persona = st.text_input("ペルソナ", value="""
    - 名前: 加藤 一郎
    - 年齢: 34歳
    - 性別: 男性
    - 職業: サラリーマン
    - 年収: 450万円
    - 地域: 東京都中央区
    - 家族構成: 妻と2歳の子供がいる
    - 趣味: 読書、ジョギング
    - 価値観: 健康的な生活を送りたい
    - 課題や悩み: 太り気味でダイエットをしたいが、忙しくて運動ができない
    """)
    
    st.write('※「ペルソナ作成」ページで作成したペルソナは以下です')
    st.code(st.session_state['persona'])
    
    # リサーチ用のプロンプト
    research_1 = """
    # Step1. ペルソナの感情と欲求の分析
    ペルソナが抱えている、 以下の要素を考慮してください:
    - 痛み、悩み、不満、不安(負の感情)
    - 快楽 喜び、 安心 (正の感情)
    - 願望、理想の未来 (強い欲求)
    具体的に、鮮明に、詳細に、独自の視点で、それぞれ5つずつ例を挙げてください。
    """
    research_2 = """
    # Step2. ペルソナの信念、欲求、感情の分析
    ペルソナの以下の要素を考慮してください:
    - Belief : 思い込み、信念
    - Desire: 望んでいること、欲求
    - Feeling: 感情 感じていること
    具体的に、鮮明に、詳細に、 独自の視点で、 それぞれ5つずつ例を挙げてください。
    """
    research_3 = f"""
    # Step3. 商品・サービスの解決策とペルソナの 目標達成・欲求分析
    {st.session_state['product_name']}を購入し、問題が解決されるとペルソナは最終的にどんな目標を達成できますか。
    また、目標を達成した場合、どのような欲求を満たしますか。
    具体的に、鮮明に、詳細に、独自の視点で、5つ例を挙げてください。
    """
    
    # リサーチしたいものを選ぶ
    st.header('以下の中から実行したいリサーチを選んでください')
    st.code(f'{research_1}')
    st.code(f'{research_2}')
    st.code(f'{research_3}')
    choose_r = st.radio(label='どれを実行しますか？',
                       options=('Step1. ペルソナの感情と欲求の分析', 
                                'Step2. ペルソナの信念、欲求、感情の分析', 
                                'Step3. 商品・サービスの解決策とペルソナの 目標達成・欲求分析'),
                       horizontal=False)
    # 変数を対応させる
    if choose_r == 'Step1. ペルソナの感情と欲求の分析':
        research = research_1
    elif choose_r == 'Step2. ペルソナの信念、欲求、感情の分析':
        research = research_2
    elif choose_r == 'Step3. 商品・サービスの解決策とペルソナの 目標達成・欲求分析':
        research = research_3


    # 実行ボタン
    execute_research = st.button('ペルソナ分析')

    # ストップウォッチ用
    stopwatch = Stopwatch()
    timer_placeholder = st.empty()

    # アウトプット用
    output = ProcessOutput()

    # 文章作成
    if execute_research:
        st.info('作成には時間がかかることがあります。気長にお待ちください')
        # 時間計測開始
        stopwatch.start()

        # 変数を受け取っておく
        product_name = st.session_state['product_name']
        product_price = st.session_state['product_price']
        product_features = st.session_state['product_features']
        
        # プロンプトを表示する
        prompt_base = make_prompt_base(product_name=product_name, 
                                     product_price=product_price, 
                                     product_features=product_features,
                                     persona=persona)
        prompt_research = make_prompt_research(base_prompt=prompt_base, research=research)
        st.header('入力するプロンプト')
        st.code(prompt_research)
        # 変数を保存
        st.session_state['made_research_prompt'] = prompt_research

        Thread(target=process_research, args=(product_name, 
                                              product_price, 
                                              product_features, 
                                              persona, research, stopwatch, output)).start()
        st.header('リサーチ結果')

    while execute_research and not output.is_set():
        timer_placeholder.text('Timer: ' + str(round(stopwatch.get_time(), 2)))
        time.sleep(0.01)

    # 結果出力
    st.write(output.get_value())
    # 変数を保存
    st.session_state['result_research'] = output.get_value()
    

# ---結果を出力---
elif selector == pagelist[2]:
    st.title("結果を出力するよ！")
    
    st.subheader("ペルソナ作成のプロンプト")
    st.code(st.session_state['made_persona_prompt'])
    st.subheader("作成されたペルソナ")
    st.code(st.session_state['persona'])
    
    st.subheader("リサーチのプロンプト")
    st.code(st.session_state['made_research_prompt'])
    st.subheader("リサーチ結果")
    st.code(st.session_state['result_research'])