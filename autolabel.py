"""Rule-based weak labeler for v1..v24 (dev + human-label calibrated).

label(rec) -> (vals: dict v->0/1, evs: dict v->str)
Evidence is always an exact substring (one line) of an original doc text.

CLI (unlabeled.csv의 자동 라벨 행을 다시 만들 때):
    python autolabel.py --out auto_labels.csv            # train_unlabeled 2만 건 전체
데이터 폴더는 이 파일이 있는 폴더(또는 환경변수 PPS_OPEN_DIR)에서 찾습니다.
"""
from __future__ import annotations
import csv, os, re
from pathlib import Path

ITEMS = [f"v{i}" for i in range(1, 25)]
ABSENCE = {"v10", "v11", "v16", "v18", "v20"}
TH = 230_000_000
LOCAL_TH = 500_000_000
ROOT = Path(os.environ.get("PPS_OPEN_DIR", Path(__file__).resolve().parent))

COMP = set()
COMP_NAMES = set()
with open(ROOT / "data/법령패키지/중기부고시/중기부고시_경쟁제품_세부품명.csv", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        c = re.sub(r"\D", "", r.get("세부품명번호") or "")
        if len(c) == 10:
            COMP.add(c)
            nm = re.sub(r"\s+", "", r.get("세부품명") or "")
            if len(nm) >= 5 and not re.search(r"기타|일반|소프트웨어|컴퓨터", nm):
                COMP_NAMES.add(nm)

SIDO = ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종", "경기", "강원", "충청북", "충청남",
        "충북", "충남", "전라북", "전북", "전라남", "전남", "경상북", "경북", "경상남", "경남", "제주"]
SIDO_CANON = {"충북": "충청북", "충남": "충청남", "전북": "전라북", "전남": "전라남", "경북": "경상북", "경남": "경상남"}
RE_SIDO = re.compile("(" + "|".join(SIDO) + r")(특별시|광역시|특별자치시|특별자치도|도|시)?")

RE_EVAL = re.compile(r"평가|배점|점수|가점|감점|심사항목|정량|정성|평점")
RE_QUAL_END = re.compile(r"(업체|자|사업자|법인)\s*(이어야|여야|로\s*제한|에\s*한|만|로서|\s*$|\.|\)|이며|일\s*것)|있어야|있는\s*(자|업체)|보유|참가\s*자격|참여\s*가능|한함|한정")


def amt_parse(s: str):
    """'3억 5천만원', '455,000,000원', '5천만원', '1억원' -> int list."""
    out = []
    for m in re.finditer(r"(\d[\d,]*(?:\.\d+)?)\s*억\s*(?:(\d[\d,]*)\s*천\s*만?)?\s*(?:(\d[\d,]*)\s*만)?\s*원?", s):
        v = float(m.group(1).replace(",", "")) * 1e8
        if m.group(2): v += int(m.group(2).replace(",", "")) * 1e7
        if m.group(3): v += int(m.group(3).replace(",", "")) * 1e4
        out.append(int(v))
    for m in re.finditer(r"(?<![\d억])(\d[\d,]*)\s*천\s*만\s*원", s):
        out.append(int(m.group(1).replace(",", "")) * 10_000_000)
    for m in re.finditer(r"(?<![\d억천])(\d{1,3}(?:,\d{3})+|\d{5,})\s*만\s*원", s):
        out.append(int(m.group(1).replace(",", "")) * 10_000)
    for m in re.finditer(r"(?<![\d.])(\d{1,3}(?:,\d{3}){2,}|\d{7,})\s*원", s):
        out.append(int(m.group(1).replace(",", "")))
    return out


class Rec:
    def __init__(self, rec):
        self.r = rec
        self.m = rec.get("meta", {}) or {}
        docs = rec.get("docs", [])
        self.ann = [d["text"] for d in docs if d.get("type") == "공고문"] or [d["text"] for d in docs[:1]]
        self.all = [d["text"] for d in docs]
        self.ann_t = "\n".join(self.ann)
        self.all_t = "\n".join(self.all)
        self.ann_lines = [l for t in self.ann for l in t.split("\n") if l.strip()]
        self.all_lines = [l for t in self.all for l in t.split("\n") if l.strip()]
        m = self.m
        a = m.get("입찰추정가격")
        if not isinstance(a, (int, float)) or a <= 0:
            b = m.get("배정예산금액")
            a = b / 1.1 if isinstance(b, (int, float)) and b > 0 else None
        self.amt = a if a and a >= 1_000_000 else None
        self.local = "지방" in str(m.get("적용계약법", ""))
        cm = str(m.get("계약방법", "")) + " " + str(m.get("낙찰방법", ""))
        self.suui = "수의" in cm
        self.nego = "협상" in cm
        self.goods = "물품" in str(m.get("업무구분", ""))
        codes = set(re.findall(r"(?<!\d)(\d{10})(?!\d)", str(m.get("세부품명번호목록") or "")))
        self.codes = codes
        self.tcodes = set(re.findall(r"(?<!\d)(\d{10})(?!\d)", self.all_t)) & COMP
        head = re.sub(r"\s+", "", self.ann_t[:800])
        self.cname = next((n for n in COMP_NAMES if n in head), None)
        title = self.ann_t[:800] + " " + str(m.get("면허업종제한목록") or "")
        self.ctitle = bool(re.search(r"행사\s*(대행|기획)|공연\s*(대행|운영)|축제\s*(기획|대행|운영)|통학\s*(운송|버스)|(건물|청사)?\s*청소\s*(용역|서비스)|경비\s*(용역|서비스|업무)|정보\s*시스템\s*(유지|개발)|(시스템|홈페이지|SW|소프트웨어)\s*(유지\s*(보수|관리)|구축|개발|고도화)", title))
        self.comp = bool(codes & COMP) or bool(self.tcodes) or bool(self.cname) or self.ctitle
        self.jo = str(m.get("조항호내용") or "")


def ev_line(line: str) -> str:
    s = line.strip()
    while s and s[0] in "=+@-":
        s = s[1:].lstrip()
    return s[:500]


def first(lines, pat, neg=None, extra=None):
    for l in lines:
        if len(l) > 1500: continue
        if re.search(pat, l) and not (neg and re.search(neg, l)) and (extra is None or extra(l)):
            return l
    return None


# ---------- shared detectors ----------
def perf_lines(R: Rec):
    out = []
    for l in R.ann_lines:
        if len(l) > 800 or "실적" not in l: continue
        if RE_EVAL.search(l): continue
        if re.search(r"실적\s*(증명|확인)서?\s*(제출|발급)|하도급|실적\s*신고|실적\s*으로\s*평가|적격|신인도", l) and not re.search(r"이상", l): continue
        if re.search(r"(실적|수행|납품|이행).{0,60}(이상|있는|보유|있어야)|(이상|보유).{0,40}실적", l):
            if amt_parse(l) or re.search(r"\d+\s*건\s*이상|실적이\s*있는|실적을\s*보유|실적\s*보유", l):
                out.append(l)
    return out


def region_lines(R: Rec):
    out = []
    for l in R.ann_lines:
        if len(l) > 900: continue
        if RE_EVAL.search(l): continue
        if re.search(r"주된\s*영업소|본점\s*소재지|본사\s*소재지|본사\(|소재지\s*를|소재지가|소재하고|관내|(본점|사업장|영업소)[^\n]{0,40}소재한|사업장을\s*소재", l) and \
           re.search(r"둔|두고|있는|있어야|소재|내에|한함|한정|업체|사업자", l) and \
           (RE_SIDO.search(l) or "[지역:" in l or "관내" in l):
            if re.search(r"제한\s*없음|지역\s*제한\s*없|전국", l): continue
            out.append(l)
    return out


def sido_set(l: str):
    s = set()
    for m in RE_SIDO.finditer(l):
        k = m.group(1)
        s.add(SIDO_CANON.get(k, k))
    return s


RE_MIDSMALL = re.compile(r"중\s*[·ㆍ․/.,∙•・]\s*소\s*기업")


RE_LAWNAME = re.compile(r"[「｢『<〈\[【][^」｣』>〉\]】]{0,60}[」｣』>〉\]】]|중소\s*기업\s*(기본법|제품|범위|청|공공구매|현황|창업|진흥|벤처|협동조합|중앙회)[^\s,.)]*|중소벤처기업부|중소기업자\s*간\s*경쟁")


def nsme(l: str) -> str:
    return RE_MIDSMALL.sub("중소기업", l)


def sme_kind(l: str):
    """returns (mentions_mid, mentions_small) after removing law/regulation names"""
    n = RE_LAWNAME.sub(" ", nsme(l))
    n = RE_LAWNAME.sub(" ", n)
    mid = bool(re.search(r"중소\s*기업|중기업", n))
    small = bool(re.search(r"(?<!중)소\s*기업|소상공인", n.replace("중소기업", "")))
    return mid, small


def only_small(l: str) -> bool:
    mid, small = sme_kind(l)
    return small and not mid


def sme_lines(R: Rec, window: bool = False):
    out = []
    L = R.all_lines
    for i, l in enumerate(L):
        if len(l) >= 900 or not re.search(r"중소기업|소기업|소상공인|중\s*[·ㆍ․/.,∙•・]\s*소\s*기업", l): continue
        w = l if (len(l) >= 150 or not window) else " ".join(L[i:i + 3])[:600]
        if RE_EVAL.search(l): continue
        if not re.search(r"확인서|소지|한함|한정|제한|업체이어야|자이어야|업체여야|참가\s*(할\s*수|가능)|입찰참가", w): continue
        if re.search(r"우대|가산|지원\s*사업|판로지원법에\s*따라\s*구매|혜택|1\s*부|협동조합|상생|해당\s*(없|되는\s*경우만)|제한입찰의\s*경우", l): continue
        out.append(l)
    return out


def label(rec):
    R = Rec(rec)
    V = {v: 0 for v in ITEMS}
    E = {v: "" for v in ITEMS}
    amt = R.amt
    th = LOCAL_TH if R.local else TH

    def put(v, line):
        V[v] = 1
        E[v] = "" if v in ABSENCE or line is None else ev_line(line)

    # v1: bidder type restricted
    l = first(R.ann_lines, r"(대학|산학협력단|협회|조합|재단|비영리\s*법인|사회적\s*(기업|협동조합)|공공기관|연구기관|학회)[^\n]{0,40}(만\s*(참여|참가|입찰)|에\s*한(함|하여|한다)|으?로\s*한정|만\s*가능|으로\s*제한)",
              neg=r"실적|평가|배점|가점|공동수급|컨소시엄|하도급|제외|아닌|우대")
    if l: put("v1", l)

    # performance family
    pl = perf_lines(R)
    rl = region_lines(R)
    if pl and amt:
        if amt < TH and not R.suui:
            put("v2", pl[0])
        for l in pl:
            vals = [x for x in amt_parse(l) if x >= 5_000_000]
            if vals and max(vals) >= amt:
                put("v3", l); break
        for l in pl + [x for x in R.ann_lines if len(x) < 600 and "실적" in x and re.search(r"있는\s*(업체|자)|있어야|보유", x) and not RE_EVAL.search(x)]:
            if re.search(r"대학|의료기관|병원|학교|어린이집|국가기관|공공기관|지방자치단체|지자체|관공서|정부투자기관|공기업|대학병원|수요기관|발주기관|교육청|군\s*부대|국공립", l) \
               and not re.search(r"민간|또는\s*법인|등\s*(에서|이)?\s*발주한\s*(모든|각종)", l):
                put("v4", l); break

    # region family
    if rl and amt:
        big = amt >= th
        if big:
            put("v5", rl[0])
        else:
            if not (R.local and R.suui):
                l6 = next((l for l in rl if "단위=기초" in l or re.search(r"[가-힣]+(시|군|구)\s*(관내|내에|에\s*(둔|소재))", l) and not RE_SIDO.search(l)), None)
                if l6: put("v6", l6)
            if not R.suui:
                l7 = next((l for l in rl if len(sido_set(l)) >= 2 and not re.search(r"인접|연접", l)), None)
                if l7: put("v7", l7)
        if pl:
            put("v8", rl[0])

    # v9: brand/model designated
    if R.goods:
        l = first(R.all_lines, r"(제조사\s*[·/및,]?\s*모델명?|모델명|제조사|브랜드)\s*[:：]\s*[A-Za-z가-힣]",
                  neg=r"동등|이상|상당|기재|입력|작성|제출|명시|표기|확인|예시|참고|또는\s*동급|해당\s*없음|제조사\s*[:：]\s*$")
        if l: put("v9", l)

    has_direct = bool(re.search(r"직접생산\s*(확인)?\s*증명서?[^\n]{0,80}(소지|보유|발급(받은|된)|있는|이어야|여야|제출)", R.all_t))
    sq = [l for l in sme_lines(R, True) if any(sme_kind(l))]
    sme_exc = bool(re.search(r"제\s*2\s*조의\s*3|우선\s*조달\s*계약[^\n]{0,20}예외|예외\s*적용", R.all_t))
    has_sme = bool(sq)
    # competitive products
    if R.comp and not R.suui:
        if not has_direct:
            put("v10", None)
        if not sq:
            put("v11", None)
        sl = [l for l in sme_lines(R) if only_small(l)]
        if sl: put("v13", sl[0])
    # v12: direct production required for a non-competitive item
    if not R.comp:
        l = first(R.ann_lines, r"직접생산\s*(확인|증명)[^\n]{0,60}(보유|소지|발급받은|있는|이어야|여야|제출)",
                  neg=r"경쟁제품|해당\s*시|해당하는\s*경우|평가|배점")
        if l: put("v12", l)

    # SME amount bands (general products)
    if amt and not R.comp:
        sl = sme_lines(R)
        osm = [l for l in sl if only_small(l)]
        mid = [l for l in sl if sme_kind(l)[0]]
        if amt >= TH:
            if sl: put("v14", sl[0])
        elif amt >= 100_000_000:
            if osm: put("v15", osm[0])
            if not has_sme and not R.suui and not sme_exc:
                put("v16", None)
        else:
            if mid and not osm and not R.suui and "중기업" not in R.jo:
                put("v17", mid[0])
            if not has_sme and not R.suui and not sme_exc:
                put("v18", None)

    # v19: supply confirmation held before bid deadline
    l = first(R.all_lines, r"확약서[^\n]{0,120}(마감\s*(일|시간)?\s*(전|이전)?까지\s*(보유|제출)|입찰서\s*(와|과)\s*함께|입찰\s*(시|전)\s*제출|보유하여야|보유해야)|확약서[^\n]{0,40}미제출\s*시\s*(입찰|참가)",
              neg=r"제출\s*가능|계약\s*(시|체결\s*시|후)\s*제출(?!.*보유)")
    if l: put("v19", l)

    # v20: SW project without large-company restriction statement
    sw = "1468" in str(R.m.get("면허업종제한목록") or "") or "소프트웨어사업자" in R.ann_t or \
        any(c.startswith(("8111", "4323")) for c in R.codes)
    if sw and not re.search(r"대기업|제48조|하한제|상호출자|중견", R.all_t):
        put("v20", None)

    # v21: joint venture min share below standard
    if re.search(r"공동수급|공동이행|공동도급", R.ann_t):
        for l in R.ann_lines:
            if "분담" in l and "공동이행" not in l: continue
            m = re.search(r"(최소\s*지분율?|지분율|출자\s*비율|참여\s*비율)[^\n]{0,20}?(\d+(?:\.\d+)?)\s*%\s*이상", l)
            if m and float(m.group(2)) < 5:
                put("v21", l); break

    # v22 / v23: briefing
    if R.nego:
        l = first(R.all_lines, r"설명회[^\n]{0,80}(미참석|불참|참석하지\s*(아니한|않은)|참석\s*(업체|자)에\s*한|참석한\s*(자|업체)(만|에\s*한))|설명회에\s*참석한\s*자",
                  neg=r"상관없이|무관|관계없이|불이익\s*없")
        if l: put("v22", l)
        if R.local:
            l = first(R.ann_lines, r"(설명회|제안요청\s*설명|과업\s*설명)[^\n]{0,80}\d{4}\s*[.\-년]\s*\d{1,2}\s*[.\-월]\s*\d{1,2}",
                      neg=r"생략|미실시|개최하지|없음")
            if l: put("v23", l)

    # v24: budget / estimated-price mismatch
    ep = R.m.get("입찰추정가격"); bg = R.m.get("배정예산금액")
    for l in R.ann_lines[:200]:
        if len(l) > 400: continue
        m = re.search(r"추정\s*가격\s*[:：]?\s*(?:금\s*)?([\d,]{7,})\s*원", l)
        if m and isinstance(ep, (int, float)) and ep > 0:
            x = int(m.group(1).replace(",", ""))
            if abs(x - ep) > 10:
                put("v24", l); break
        m = re.search(r"(배정\s*예산|사업\s*예산|소요\s*예산)\s*(액|금액)?\s*[:：]?\s*(?:금\s*)?([\d,]{7,})\s*원", l)
        if m and isinstance(bg, (int, float)) and bg > 0:
            x = int(m.group(3).replace(",", ""))
            if abs(x - bg) > 10 and abs(x - bg * 1.1) > 10 and abs(x * 1.1 - bg) > 10:
                put("v24", l); break
    return V, E


def main():
    import argparse, gzip, json
    ap = argparse.ArgumentParser(description="train_unlabeled 전체에 규칙 기반 자동 라벨을 붙여 49열 CSV로 저장")
    ap.add_argument("--input", default=str(ROOT / "train_unlabeled.jsonl.gz"))
    ap.add_argument("--out", default="auto_labels.csv")
    ap.add_argument("--limit", type=int, default=None)
    x = ap.parse_args()
    hdr = ["id"] + ITEMS + [f"e{i}" for i in range(1, 25)]
    n = 0
    with gzip.open(x.input, "rt", encoding="utf-8") as f, open(x.out, "w", encoding="utf-8", newline="") as g:
        w = csv.writer(g); w.writerow(hdr)
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            V, E = label(rec)
            w.writerow([rec["id"]] + [V[v] for v in ITEMS] + [E[v] if V[v] else "" for v in ITEMS])
            n += 1
            if x.limit and n >= x.limit:
                break
    print("rows", n, "->", x.out)


if __name__ == "__main__":
    main()
