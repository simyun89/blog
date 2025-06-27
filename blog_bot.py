import os, re, time, requests, pandas as pd
import textwrap, json, tiktoken
from datetime import datetime, timedelta
from collections import Counter
from konlpy.tag import Okt
import base64
import openai

confluence_api_token = os.environ.get("CONFLUENCE_API_TOKEN")
confluence_api_user = os.environ.get("CONFLUENCE_API_USER")
CID  = os.environ.get("NAVER_CLIENT_ID")
CSEC = os.environ.get("NAVER_CLIENT_SECRET")

openai_api_key = os.environ.get("OPENAI_API_KEY")
client = openai.OpenAI(api_key=openai_api_key)

HEAD = {'X-Naver-Client-Id': CID, 'X-Naver-Client-Secret': CSEC}
URL  = 'https://openapi.naver.com/v1/search/blog.json'

space_key = 'CSO'
parent_page_id = '661848065'
title = f"주간 블로그 모니터링 리포트_{datetime.now():%Y-%m-%d}"
keywords = [ '"이즐 교통카드"', '"ezl"',  '"이즐"', '"캐시비"', '"이동의즐거움"', '"티머니"']   # 원하는 만큼 추가
mapping  = {'이동의즐거움':'자사', '이즐':'자사', 'ezl':'자사', '이즐 교통카드':'자사', '캐시비':'자사',
            '티머니':'경쟁사'}
end_date   = (datetime.now() - timedelta(days=1)).date()
start_date = (datetime.now() - timedelta(days=7)).date()
kw_order = [kw.strip('"') for kw in keywords]
okt = Okt() 

# 블로그 데이터 수집
detail_rows = []
for kw in keywords:
    kw_clean = kw.strip('"')
    start = 1
    while start <= 1000:
        js = requests.get(URL,
                          params={'query': kw, 'display':100,
                                  'start':start, 'sort':'date'},
                          headers=HEAD).json()
        if 'items' not in js: break

        for it in js['items']:
            pdate = datetime.strptime(it['postdate'], '%Y%m%d').date()
            if pdate < start_date:
                start = 1001; break
            if pdate > end_date:
                continue
            detail_rows.append({
                '구분'   : mapping[kw_clean],
                '키워드' : kw_clean,
                '날짜'   : pdate.strftime('%Y-%m-%d'),
                '제목'   : re.sub('<.*?>', '', it['title']),
                'URL'    : it['link']
            })

        if js.get('display', 0) < 100 or start >= 1000:
            break
        start += 100

detail_df = (pd.DataFrame(detail_rows)
             .drop_duplicates('URL')         # (URL 중복 제거) 여러 URL 중복이 있을 시 1건만 남겨둠 키워드 순으로 남겨둠
             .reset_index(drop=True))

# ── 2. 요약 테이블 ────────────────────────────────────────
# 2-a) 구분별 요약
summary_df = (detail_df.groupby('구분', observed=True)
                        .size()
                        .reset_index(name='최근 7일')
                        .sort_values('구분'))

# 2-b) 구분별 x 키워드 요약
summary_kw_df = (detail_df.assign(
                     키워드=pd.Categorical(detail_df['키워드'],
                                           categories=kw_order, ordered=True))
                 .groupby(['구분','키워드'], observed=True)
                 .size()
                 .reset_index(name='최근 7일')
                 .sort_values(['구분','키워드']))

# 2-c) 일자별 x 키워드별 건수
daily_df = (detail_df.groupby(['날짜','키워드'])
                      .size()
                      .unstack(fill_value=0)
                      .reindex(columns=kw_order, fill_value=0)
                      .reindex(pd.date_range(start_date, end_date)
                                  .strftime('%Y-%m-%d'), fill_value=0))


# 2-d) 단어 빈도
stop = {'이즐', 'ezl', '티머니', '캐시비'} # 제외할 단어
ja_cn, co_cn = Counter(), Counter()
for row in detail_df[['구분', '제목']].itertuples(index=False):
    tokens = [w for w, pos in okt.pos(row.제목, stem=True)
              if pos in ('Noun','Adjective') and len(w) > 1 and w not in stop]
    (ja_cn if row.구분 == '자사' else co_cn).update(tokens)

N = 20                                    # Top N 추출
ja_top = ja_cn.most_common(N)
co_top = co_cn.most_common(N)
max_len = max(len(ja_top), len(co_top))
ja_top += [('', 0)] * (max_len - len(ja_top))
co_top += [('', 0)] * (max_len - len(co_top))
freq_side_df = pd.DataFrame({
    '단어(자사)' : [w for w, _ in ja_top],
    '건수(자사)'  : [c for _, c in ja_top],
    '단어(경쟁사)': [w for w, _ in co_top],
    '건수(경쟁사)' : [c for _, c in co_top],
})

# 2-e) GPT로 이슈 클러스터링
enc = tiktoken.encoding_for_model("gpt-4o-mini")
def num_tokens(txt): return len(enc.encode(txt))

def gpt_cluster(title_list, label):
    titles_block = "\n".join(f"- {t}" for t in title_list)
    if num_tokens(titles_block) > 7000:
        titles_block = "\n".join(f"- {t}" for t in title_list[:600])

    prompt = textwrap.dedent(f"""
      아래는 최근 7일간 **{label}** 블로그 글 제목 리스트입니다.

      {titles_block}

      다음 조건을 반드시 지켜주세요.
      1) 의미가 유사하거나 반복되는 제목끼리 묶어서, '이슈 유형'을 최대 10개까지 도출하세요.
      2) 각 이슈 유형별로 제목 건수를 세서, ["이슈 유형", 건수] 형태의 JSON 배열로 출력하세요.
      3) 반드시 다음과 같은 필드명만 사용하세요: "type" (이슈 유형), "count" (건수)
      4) 설명, 해설, 추가 코멘트, 예시 등은 절대 포함하지 마세요. 오직 JSON만 반환하세요.
      
      오직 JSON 데이터만 결과로 출력하세요.
    """).strip()

    resp = openai.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.I|re.M).strip()
    raw = raw[raw.find('['): raw.rfind(']') + 1]   # '[' ... ']' 사이만

    issue_df = pd.DataFrame(json.loads(raw))
    issue_df.columns = [c.lower() for c in issue_df.columns]

    if 'count' not in issue_df.columns:
        num_cols = [c for c in issue_df.columns if pd.api.types.is_numeric_dtype(issue_df[c])]
        if num_cols:
            issue_df = issue_df.rename(columns={num_cols[0]: 'count'})

    if 'type' not in issue_df.columns:
        cat_cols = [c for c in issue_df.columns if not pd.api.types.is_numeric_dtype(issue_df[c]) and c != '구분']
        if cat_cols:
            issue_df = issue_df.rename(columns={cat_cols[0]: 'type'})
    issue_df.insert(0, '구분', label)   # 자사/경쟁사 표시
    return issue_df

titles_jasa = detail_df.query("구분 == '자사'")['제목'].tolist()   #  자사·경쟁사 각각 호출
titles_comp = detail_df.query("구분 == '경쟁사'")['제목'].tolist()

issue_df_jasa = gpt_cluster(titles_jasa, '자사')
issue_df_comp = gpt_cluster(titles_comp, '경쟁사')

issue_df_all = pd.concat([issue_df_jasa, issue_df_comp], ignore_index=True)

def make_bullet(df):     #  리포트 문구
    return "\n".join(
        f"- **{row['type']}** : {row['count']}건"
        for _, row in df.sort_values('count', ascending=False).iterrows()
    )
print("\n[자사 이슈 Top]\n", make_bullet(issue_df_jasa))
print("\n[경쟁사 이슈 Top]\n", make_bullet(issue_df_comp))


# 결과 확인 / 저장
pd.set_option('display.max_colwidth', None)
detail_df.to_csv('blog_detail.csv',  index=False, encoding='utf-8-sig')

# ── 3. 리포트용 HTML · 본문 구성 ──────────────────────────────
summary_html      = summary_df.fillna('').to_html(index=False, border=1)
summary_kw_html   = summary_kw_df.to_html(index=False, border=1)    # fillna('') X
daily_html        = daily_df.fillna('').reset_index(names='날짜').to_html(index=False, border=1)
freq_html         = freq_side_df.fillna('').to_html(index=False, border=1)
issue_jasa_tbl_html = (
    issue_df_jasa.fillna('').rename(columns={'type': '이슈 유형', 'count': '건수'})
                 .to_html(index=False, border=1)
)
issue_comp_tbl_html = (
    issue_df_comp.fillna('').rename(columns={'type': '이슈 유형', 'count': '건수'})
                 .to_html(index=False, border=1)
)

# 3-b) Confluence 본문 템플릿
body_html = f"""
<br><h2>📝 참고 사항</h2>
<ul>
  <li>본 보고서는 매주 월요일 오전 7시에 자동 발송됩니다.</li>
  <li>지난주(월~일) 네이버 블로그에 게시된 글을 기준으로 작성되었습니다.</li>
  <li>분석에 사용된 검색 키워드 : {' , '.join(kw_order)}</li>
  <li>동일한 URL은 중복을 제거하여 집계하였습니다.</li>
</ul>

<br><h2>📊 1. 구분별 블로그 언급 건수</h2>
{summary_html}

<br><h2>📑 2. 구분 × 키워드별 언급 건수</h2>
{summary_kw_html}

<br><h2>📈 3. 일자별 키워드 트렌드</h2>
{daily_html}

<br><h2>🗣️ 4. 단어 빈도(Top 20)</h2>
{freq_html}

<br><h2>💡 5. AI 기반 블로그 이슈 유형 정리</h2>
<ul>
  <li>GPT로 블로그 글을 분석하여 주요 이슈를 유형별로 정리한 표입니다.</li>
</ul>
<h3>① 자사 TOP 이슈</h3>
{issue_jasa_tbl_html}
<br><h3>② 경쟁사 TOP 이슈</h3>
{issue_comp_tbl_html}

<br><h2>📥 [Raw Data] 블로그 상세 내역 다운로드</h2>
"""

# ── 4. Confluence 페이지 생성 ────────────────────────────────
confluence_domain = 'myezl.atlassian.net'
headers = {
    'Authorization': 'Basic ' +
        base64.b64encode(f'{confluence_api_user}:{confluence_api_token}'
                         .encode()).decode(),
    'Content-Type': 'application/json'
}
base_url = f'https://{confluence_domain}/wiki/rest/api/content/'

page_data = {
    "type": "page",
    "title": title,
    "ancestors": [{"id": parent_page_id}],
    "space": {"key": space_key},
    "body": {"storage": {"value": body_html, "representation": "storage"}}
}
resp = requests.post(base_url, headers=headers, json=page_data)
resp.raise_for_status()
page_id = resp.json()['id']
print("페이지 생성 ✔", page_id)

# ── 5. CSV 첨부 & 링크 삽입 ──────────────────────────────────
attach_url  = f"{base_url}{page_id}/child/attachment"
attach_head = {
    "Authorization": headers['Authorization'],
    "X-Atlassian-Token": "no-check",
}
with open('blog_detail.csv', 'rb') as fp:
    files = {"file": ("blog_detail.csv", fp, "text/csv")}
    aresp = requests.post(attach_url, headers=attach_head, files=files)
aresp.raise_for_status()
fname = aresp.json()['results'][0]['title']
print("첨부 ✔", fname)

# 페이지 버전 올려서 첨부 링크(ri:attachment) 삽입
ver = requests.get(f"{base_url}{page_id}?expand=version",
                   headers=headers).json()['version']['number']
patch = {
    "version": {"number": ver + 1, "minorEdit": True},
    "title"  : page_data["title"],
    "type"   : "page",
    "body"   : {
        "storage": {
            "value": body_html + f'<p><ac:link><ri:attachment '
                      f'ri:filename="{fname}" /></ac:link></p>',
            "representation": "storage"
        }
    }
}
requests.put(f"{base_url}{page_id}", headers=headers, json=patch
             ).raise_for_status()
print("본문에 첨부 링크 추가 ✔")



