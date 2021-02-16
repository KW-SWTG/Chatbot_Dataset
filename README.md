생활법령 데이터셋
=================

## 1. 생활법령이란?
찾기 쉬운 생활법령정보는 법제처에서 국민이 실생활에 필요한 법령을 쉽게 찾아보고 이해할 수 있도록 관련 법령정보를 국민의 생활중심으로 재분류하여 제공하고 있습니다.
노인도우미 챗봇 프로젝트에서는 MRC 기계독해를 위하여 "노인복지", "치매노인", "고령자 고용" 생활법령을 엑셀 파일로 정제하였습니다.
<br>

홈페이지 : [찾기쉬운 생활법령정보](https://www.easylaw.go.kr/CSP/Main.laf)
![img1](https://github.com/KW-SWTG/Chatbot_Dataset/blob/master/img/img1.png)

<br>
<br><br>

## 2. Test을 위한 데이터셋 
"cate1"(1차 카테고리), "Q"(질문), "A"(응답), "paragraph"(문단) 4개의 column으로 구성되었습니다.

<br>
<br><br>

## 3. 모든 1차 카테고리
![img6](https://github.com/KW-SWTG/Chatbot_Dataset/blob/master/img/img6.png)

<br>
category = <br>
{ <br>
    '창업': ['결혼중개업', '경비(업)', '귀농인', '네일샵 창업ㆍ운영', '동업계약', '메이크업샵 창업ㆍ운영', '미용실 창업ㆍ운영', '민간임대주택사업자', '민박 사업자', '반찬가게 창업ㆍ운영', '세탁소 운영자', '소상공인 지원', '음식점 운영', '음식점 창업', '인터넷쇼핑몰 창업자', '청소(업)', '체육시설 설치ㆍ운영', '커피전문점 창업ㆍ운영', '펜션 사업자', '프랜차이즈(가맹계약)', '피부관리실 창업ㆍ운영', '학원의 설립ㆍ운영 및 과외교습'], <br>
    '소비자': ['가공식품', '건강기능식품', '농축수산물', '다단계판매', '물', '소비자 분쟁해결', '소비자 안전정보', '인터넷 쇼핑', '택배', '화장품'], <br>
    '국방&보훈': ['여군', '대체역ㆍ보충역', '준비역ㆍ현역', '예비군 및 민방위'],<br>
    '교통&운전': ['개인택시운전', '교통사고', '안전한 출퇴근', '부설주차장 설치ㆍ운영', '자동차 구입ㆍ관리', '자동차 운전면허', '자전거 운전자', '중고차 매매', '화물자동차 운송사업'],<br>
    '가정법률' : ['가족관계 등록', '상속ㆍ유언', '재혼ㆍ이혼ㆍ결혼준비자', '입양', '장사(장례ㆍ매장ㆍ화장ㆍ자연장)', '태아 및 신생아', '후견제도'],<br>
    '문화&여가생활' : ['면세점 이용', '반려동물과 생활하기', '저작권보호', '캠핑(야영)', '해외여행자'],<br>
    '무역&출입국' : ['수출입 무역제도', '비자,여권,국적', '수출입 검역', '외국인투자자'],<br>
    '사회안전&범죄' : ['가정폭력 피해자', '과태료 납부자', '무고죄 피해자ㆍ가해자', '범죄피해자', '부정청탁 및 금품수수 금지', '성범죄 피해자', '성희롱 피해자', '소방안전관리', '응급의료', '인권침해', '전자금융범죄', '집회ㆍ시위자', '폭행ㆍ상해의 피해자ㆍ가해자'],<br>
    '아동청소년&교육' : ['가수(아이돌)', '근로청소년', '불량식품', '실종아동', '아동ㆍ청소년 대상 성범죄', '아동학대', '어린이 생활건강', '어린이 생활안전', '어린이 식품안전', '어린이집 설치ㆍ운영', '영유아', '외국인유학생', '청소년유해환경', '청소년의 인터넷 이용하기', '학교 밖 청소년', '학교폭력', '해외유학자'],<br>
    '금융&금전' : ['금융투자자(펀드)', '금전거래', '대부업체(사채) 이용자', '보증', '보험계약자', '보험업종사자', '신용카드 이용자', '은행예금자', '주택연금'],<br>
    '근로&노동' : ['건설일용 근로자', '고객응대 근로자', '고령자 고용', '기간제 및 단시간근로자', '산업재해보상보험', '시간선택제 근로자', '실업급여', '여성근로자', '외국인근로자 고용취업', '유연근무제', '일과 가정생활', '임금', '장애인 고용', '퇴직급여제도', '파견근로자', '해고근로자'],<br>
    '부동산&임대차' : ['건물관리', '건축법 등 위반건축물(불법건축물)', '공공임대주택 입주자', '공인중개사', '농지', '단독주택건축(신축ㆍ개축ㆍ증축ㆍ대수선)', '부동산', '산지전용', '상가건물 임대차', '아파트', '용도변경', '이사', '재개발사업', '주택임대차'], <br>
    '사업' : ['ICT 규제샌드박스', '가족친화기업', '공장 설립', '규제자유특구(지역단위 규제샌드박스)', '금융규제 샌드박스', '비영리 사단법인', '비영리 재단법인', '사회적기업', '산업융합 규제샌드박스', '옥외광고물 설치자', '유한책임회사(설립ㆍ운영)', '유한회사(설립ㆍ운영)', '주식회사 설립', '중소ㆍ벤처기업 창업', '합명회사(설립ㆍ운영)', '합자회사(설립ㆍ운영)', '협동조합(설립ㆍ운영)'],<br>
    '정보통신&기술' : ['개인정보보호', '인터넷 명예훼손', '인터넷 불법이용 규제', '특허권', '휴대전화 이용자'],<br>
    '복지' : ['1인 가구', '감염병 예방 및 관리', '결혼이민자ㆍ재외동포', '국민건강보험', '금연', '기부 나눔', '기초생활보장', '긴급복지지원', '노인복지', '다문화가족', '다자녀가구', '북한이탈주민', '신혼부부', '암', '임산부', '장기기증ㆍ이식', '장애인 교육ㆍ복지', '치매 노인', '한부모가족'],<br>
    '민형사&소송' : ['가압류 신청', '가처분 신청', '개인파산ㆍ면책절차', '개인회생절차', '공탁', '국민참여재판', '나홀로 민사소송', '소액사건재판', '의료분쟁', '행정소송', '행정심판'],<br> 
    '국가 및 지자체' : ['공유재산 이용자', '국가 공사계약자', '국유재산 이용자', '선거권자(유권자)', '지방자치단체', '주민의 권리', '청원ㆍ민원'],<br>
    '환경&에너지' : ['1회용품 줄이기', '가정에너지 절약', '대기오염', '산업폐수', '미세먼지', '소음ㆍ진동', '실내공기질 관리', '이웃 간 분쟁 해결', '자동차 배출가스 규제', '자연재해', '자원재활용', '폐기물처리(업)', '환경분쟁 해결', '환경친화적 자동차', '환경표지인증']<br>
}<br>
