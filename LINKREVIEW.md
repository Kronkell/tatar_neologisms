# LinkReview

Обзор литературы к работе «Сколько попарных сравнений заменяют признаки объектов: априорный прогноз для иерархической модели Брэдли–Терри». Работы сгруппированы по темам; в последнем столбце — чем работа полезна нам. **Базовые методы** для сравнения: модель Брэдли–Терри без признаков [1–4] и модель «только признаки» [6, 37].

## Модели попарных сравнений

| # | Тема | Название | Год | Авторы | Ссылка | Чем полезно |
|---|---|---|---|---|---|---|
| 1 | Базовая модель | Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons | 1952 | R. A. Bradley, M. E. Terry | [Biometrika](https://doi.org/10.2307/2334029) | Исходная модель P(i≻j)=σ(u_i−u_j); наш базовый метод. |
| 2 | Базовая модель | A Law of Comparative Judgment | 1927 | L. L. Thurstone | [Psychol. Review](https://doi.org/10.1037/h0070288) | Альтернативная (пробит) функция связи; показывает, что выбор связи не принципиален. |
| 3 | Базовая модель | Individual Choice Behavior | 1959 | R. D. Luce | книга, Wiley | Аксиома выбора, обобщение BT на выбор из многих. |
| 4 | Алгоритмы | MM Algorithms for Generalized Bradley–Terry Models | 2004 | D. R. Hunter | [Ann. Stat.](https://doi.org/10.1214/aos/1079120141) | Сходящиеся алгоритмы ММП для BT; альтернатива нашему L-BFGS. |
| 5 | Обзор | Models for Paired Comparison Data: A Review with Emphasis on Dependent Data | 2012 | M. Cattelan | [Stat. Science](https://doi.org/10.1214/12-STS396) | Обзор расширений BT, в том числе с ковариатами и зависимостью; помогает позиционировать модель. |
| 6 | Признаки | Response Surface Fitting Using a Generalization of the Bradley–Terry Paired Comparison Model | 1973 | A. Springall | Applied Statistics 22(1) | Первая BT с полезностью как функцией признаков — модель «только признаки». |
| 7 | Признаки | Bradley–Terry Models in R: The BradleyTerry2 Package | 2012 | H. Turner, D. Firth | [JSS](https://doi.org/10.18637/jss.v048.i09) | Структурированная BT с ковариатами и случайным эффектом объекта — это и есть наша иерархическая модель. |

## Байесовские модели предпочтений и априорная информация

| # | Тема | Название | Год | Авторы | Ссылка | Чем полезно |
|---|---|---|---|---|---|---|
| 8 | GP | Preference Learning with Gaussian Processes | 2005 | W. Chu, Z. Ghahramani | [ICML](https://doi.org/10.1145/1102351.1102369) | Байесовская модель предпочтений с признаками и лапласовым приближением; наша модель — её линейный частный случай. Нет правила объёма опроса. |
| 9 | GP | Collaborative Gaussian Processes for Preference Learning | 2012 | N. Houlsby et al. | NeurIPS | Обобщение на многих пользователей; идея общих и индивидуальных предпочтений. |
| 10 | Рейтинги | TrueSkill: A Bayesian Skill Rating System | 2007 | R. Herbrich, T. Minka, T. Graepel | NeurIPS | Последовательное байесовское обновление полезностей по сравнениям. |
| 11 | Вывод | Efficient Bayesian Inference for Generalized Bradley–Terry Models | 2012 | F. Caron, A. Doucet | [JCGS](https://doi.org/10.1080/10618600.2012.638220), [arXiv](https://arxiv.org/abs/1011.1761) | Точный байесовский вывод (сэмплирование) для BT; путь к проверке лапласова приближения на разреженных планах. |
| 12 | Активное обучение | Bayesian Active Learning for Classification and Preference Learning | 2011 | N. Houlsby et al. | [arXiv](https://arxiv.org/abs/1112.5745) | Критерий BALD для выбора информативных сравнений. |
| 13 | Приближение | Accurate Approximations for Posterior Moments and Marginal Densities | 1986 | L. Tierney, J. B. Kadane | [JASA](https://doi.org/10.1080/01621459.1986.10478240) | Обоснование лапласова приближения, на котором построены теоремы. |
| 14 | Априорная информация | Determining the Effective Sample Size of a Parametric Prior | 2008 | S. Morita, P. F. Thall, P. Müller | [Biometrics](https://doi.org/10.1111/j.1541-0420.2007.00888.x) | Эффективный размер выборки априорного распределения — идейный предок нашей «экономии сравнений». |

## Сколько сравнений нужно: топология графа сравнений

| # | Тема | Название | Год | Авторы | Ссылка | Чем полезно |
|---|---|---|---|---|---|---|
| 15 | Спектральные методы | Rank Centrality: Ranking from Pairwise Comparisons | 2017 | S. Negahban, S. Oh, D. Shah | [Oper. Research](https://doi.org/10.1287/opre.2016.1534) | Ошибка ранжирования через спектральную щель графа сравнений. |
| 16 | Минимакс | Estimation from Pairwise Comparisons: Sharp Minimax Bounds with Topology Dependence | 2016 | N. B. Shah et al. | [JMLR](https://arxiv.org/abs/1505.01462) | Ошибка BT определяется спектром лапласиана; наша теорема 2 — аналог с априорной точностью. |
| 17 | Минимакс | Minimax-Optimal Inference from Partial Rankings | 2014 | B. Hajek, S. Oh, J. Xu | NeurIPS | Оптимальные оценки для частичных ранжирований и роль графа. |
| 18 | Top-K | Spectral MLE: Top-K Rank Aggregation from Pairwise Comparisons | 2015 | Y. Chen, C. Suh | ICML | Поиск лучших K объектов; связь с задачей выбора лучшей замены. |
| 19 | Top-K | Spectral Method and Regularized MLE Are Both Optimal for Top-K Ranking | 2019 | Y. Chen, J. Fan, C. Ma, K. Wang | [Ann. Stat.](https://doi.org/10.1214/18-AOS1745) | Оптимальность регуляризованного ММП — обоснование нашего MAP-критерия. |
| 20 | Непараметрика | Simple, Robust and Optimal Ranking from Pairwise Comparisons | 2018 | N. B. Shah, M. J. Wainwright | JMLR | Подсчёт побед без параметрической модели; ограничение параметрического подхода. |
| 21 | План опроса | Graph Resistance and Learning from Pairwise Comparisons | 2019 | J. Hendrickx, A. Olshevsky, V. Saligrama | [ICML / arXiv](https://arxiv.org/abs/1902.00141) | Ошибка через эффективное сопротивление графа и выбор плана без адаптивности; база для эксперимента с планами. |
| 22 | Неопределённость | Uncertainty Quantification in the Bradley–Terry–Luce Model | 2023 | C. Gao, Y. Shen, A. Y. Zhang | [arXiv](https://arxiv.org/abs/2110.03874) | Асимптотическая нормальность оценок BT — обоснование гауссовского приближения. |

## Выбор пар и активное ранжирование

| # | Тема | Название | Год | Авторы | Ссылка | Чем полезно |
|---|---|---|---|---|---|---|
| 23 | Активное ранжирование | Active Ranking Using Pairwise Comparisons | 2011 | K. G. Jamieson, R. D. Nowak | NeurIPS | Адаптивный выбор пар сокращает опрос; у нас план статичен. |
| 24 | Активное ранжирование | Active Ranking from Pairwise Comparisons and When Parametric Assumptions Do Not Help | 2019 | R. Heckel et al. | [Ann. Stat.](https://doi.org/10.1214/18-AOS1772) | Пределы выигрыша от параметрических допущений при активном опросе. |
| 25 | Краудсорсинг | Pairwise Ranking Aggregation in a Crowdsourced Setting | 2013 | X. Chen et al. | [WSDM](https://doi.org/10.1145/2433396.2433420) | BT с учётом качества исполнителей и активный выбор пар в краудсорсинге. |
| 26 | Активное обучение | Just Sort It! A Simple and Effective Approach to Active Preference Learning | 2017 | L. Maystre, M. Grossglauser | [ICML](https://proceedings.mlr.press/v70/maystre17a.html) | Простая адаптивная схема (сортировка с шумом), сильный практический базовый метод. |
| 27 | Бандиты | The K-armed Dueling Bandits Problem | 2012 | Y. Yue et al. | [JCSS](https://doi.org/10.1016/j.jcss.2011.12.028) | Постановка поиска лучшего объекта по сравнениям — следующий шаг работы (top-1 вместо полного порядка). |
| 28 | Бандиты | Preference-based Online Learning with Dueling Bandits: A Survey | 2021 | V. Bengs et al. | [JMLR](https://jmlr.org/papers/v22/18-546.html) | Обзор дуэльных бандитов; ориентир для продолжения. |
| 29 | Планирование эксперимента | Bayesian Experimental Design: A Review | 1995 | K. Chaloner, I. Verdinelli | [Stat. Science](https://doi.org/10.1214/ss/1177009939) | A-оптимальность (след апостериорной ковариации) — наш риск R. |

## Ответы людей и предсказания моделей

| # | Тема | Название | Год | Авторы | Ссылка | Чем полезно |
|---|---|---|---|---|---|---|
| 30 | PPI | Prediction-Powered Inference | 2023 | A. N. Angelopoulos et al. | [Science](https://doi.org/10.1126/science.adi6000) | Несмещённое объединение немногих меток с предсказаниями; альтернатива байесовской подсказке. |
| 31 | PPI | PPI++: Efficient Prediction-Powered Inference | 2023 | A. N. Angelopoulos, J. C. Duchi, T. Zrnic | [arXiv](https://arxiv.org/abs/2311.01453) | Оптимальный вес предсказаний; выигрыш ограничен корреляцией — сравнение с нашей ρ. |
| 32 | Оценка моделей | AutoEval Done Right: Using Synthetic Data for Model Evaluation | 2025 | P. Boyeau et al. | [ICML / arXiv](https://arxiv.org/abs/2403.07008) | PPI для оценки моделей, включая попарные сравнения. |
| 33 | Ранжирование | Prediction-Powered Ranking of Large Language Models | 2024 | I. Chatzi et al. | [NeurIPS / arXiv](https://arxiv.org/abs/2402.17826) | Ближайшая работа: люди и LLM-судья в попарном ранжировании; несмещённо, но без формулы объёма опроса. |
| 34 | Активный вывод | Active Statistical Inference | 2024 | T. Zrnic, E. J. Candès | [ICML / arXiv](https://arxiv.org/abs/2403.03208) | Выбор объектов для разметки по неопределённости модели. |
| 35 | LLM-судья | Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena | 2023 | L. Zheng et al. | NeurIPS D&B | Согласие LLM-судьи с людьми, смещения судьи; возможная «подсказка» в нашей модели. |
| 36 | Попарные оценки | Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference | 2024 | W.-L. Chiang et al. | ICML | Крупнейшее применение BT к человеческим сравнениям. |
| 37 | RLHF | Deep Reinforcement Learning from Human Preferences | 2017 | P. F. Christiano et al. | NeurIPS | BT с нейросетевыми признаками как модель вознаграждения. |
| 38 | RLHF | Training Language Models to Follow Instructions with Human Feedback | 2022 | L. Ouyang et al. | NeurIPS | То же в масштабе LLM; объём человеческих сравнений — главная статья расходов. |
| 39 | LLM-аннотаторы | ChatGPT Outperforms Crowd Workers for Text-Annotation Tasks | 2023 | F. Gilardi, M. Alizadeh, M. Kubli | [PNAS](https://doi.org/10.1073/pnas.2305016120) | Аргумент за LLM как дешёвую подсказку. |
| 40 | Многоязычие | Are Large Language Model-based Evaluators the Solution to Scaling Up Multilingual Evaluation? | 2024 | R. Hada et al. | [EACL Findings](https://aclanthology.org/2024.findings-eacl.71) | Для неанглийских языков LLM-оценки нужно калибровать по носителям — мотивация для татарского. |

## Разметка толпой

| # | Тема | Название | Год | Авторы | Ссылка | Чем полезно |
|---|---|---|---|---|---|---|
| 41 | Агрегация | Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm | 1979 | A. P. Dawid, A. M. Skene | [Applied Statistics](https://doi.org/10.2307/2346806) | Модель качества аннотаторов; возможное расширение на неоднородных носителей. |
| 42 | Краудсорсинг | Cheap and Fast — But is it Good? Evaluating Non-Expert Annotations for Natural Language Tasks | 2008 | R. Snow et al. | EMNLP | Сколько непрофессиональных разметчиков заменяют эксперта — родственный вопрос. |
| 43 | Шкалы | Best–Worst Scaling More Reliable than Rating Scales | 2017 | S. Kiritchenko, S. M. Mohammad | ACL | Сравнительные оценки надёжнее шкал — обоснование попарного опроса. |

## Лексическое упрощение и сложность слов (данные эксперимента)

| # | Тема | Название | Год | Авторы | Ссылка | Чем полезно |
|---|---|---|---|---|---|---|
| 44 | Данные | A Word-Complexity Lexicon and A Neural Readability Ranking Model for Lexical Simplification | 2018 | M. Maddela, W. Xu | [arXiv](https://arxiv.org/abs/1810.05754), [GitHub](https://github.com/mounicam/lexical_simplification) | Оценки сложности 15 тыс. слов от 11 человек — наши реальные сравнения. |
| 45 | Данные | Benchmarking Lexical Simplification Systems | 2016 | G. Paetzold, L. Specia | [LREC](https://aclanthology.org/L16-1491/) | Набор BenchLS: слово и кандидаты на замену — наши группы. |
| 46 | Данные | SemEval-2012 Task 1: English Lexical Simplification | 2012 | L. Specia, S. K. Jauhar, R. Mihalcea | SemEval | Ранжирование замен по простоте людьми — исходная постановка. |
| 47 | Методы | Lexical Simplification with Pretrained Encoders | 2020 | J. Qiang et al. | [AAAI](https://doi.org/10.1609/aaai.v34i05.6389), [GitHub](https://github.com/qiang2100/BERT-LS) | Генерация кандидатов языковой моделью; источник версии BenchLS. |
| 48 | Обзор | A Survey on Lexical Simplification | 2017 | G. H. Paetzold, L. Specia | [JAIR](https://doi.org/10.1613/jair.5526) | Признаки простоты слова (частота, длина) — наши признаки. |
| 49 | Данные | CompLex — A New Corpus for Lexical Complexity Prediction from Likert Scale Data | 2020 | M. Shardlow, M. Cooper, M. Zampieri | [arXiv](https://arxiv.org/abs/2003.07008) | Альтернативный набор оценок сложности для проверки переносимости. |
| 50 | Многоязычие | Findings of the TSAR-2022 Shared Task on Multilingual Lexical Simplification | 2022 | H. Saggion et al. | [arXiv](https://arxiv.org/abs/2302.02888) | Замены слов на английском, испанском, португальском — данные для других языков. |

## Неологизмы, правдоподобие слов, малоресурсные языки

| # | Тема | Название | Год | Авторы | Ссылка | Чем полезно |
|---|---|---|---|---|---|---|
| 51 | Неологизмы | Where New Words Are Born: Distributional Semantic Analysis of Neologisms and Their Semantic Neighborhoods | 2020 | M. Ryskina et al. | [arXiv](https://arxiv.org/abs/2001.07740) | Какие факторы определяют появление новых слов — кандидаты в признаки. |
| 52 | Правдоподобие слов | Determinants of Wordlikeness: Phonotactics or Lexical Neighborhoods? | 2001 | T. M. Bailey, U. Hahn | [JML](https://doi.org/10.1006/jmla.2000.2756) | Обоснование признака «похожесть на слова языка» (символьная LM). |
| 53 | Фонотактика | A Maximum Entropy Model of Phonotactics and Phonotactic Learning | 2008 | B. Hayes, C. Wilson | [Linguistic Inquiry](https://doi.org/10.1162/ling.2008.39.3.379) | Модель приемлемости форм — возможный признак для татарского (сингармонизм). |
| 54 | Малоресурсные языки | The State and Fate of Linguistic Diversity and Inclusion in the NLP World | 2020 | P. Joshi et al. | ACL | Классификация языков по ресурсам; татарский — малоресурсный. |
| 55 | Татарский | Finite-State Morphological Transducers for Three Kypchak Languages | 2014 | J. Washington, I. Salimzyanov, F. Tyers | [LREC](https://aclanthology.org/L14-1143/) | Морфологический анализатор татарского (Apertium) — фильтр допустимых замен. |
| 56 | Тюркские языки | TUMLU: A Unified and Native Language Understanding Benchmark for Turkic Languages | 2025 | J. Isbarov et al. | [arXiv](https://arxiv.org/abs/2502.11020) | Насколько LLM понимают татарский — качество LLM-подсказки. |
| 57 | Заимствования | Loanwords in the World's Languages: A Comparative Handbook | 2009 | M. Haspelmath, U. Tadmor (ред.) | [De Gruyter](https://doi.org/10.1515/9783110218442) | Лингвистический контекст: как языки заимствуют и замещают слова. |
