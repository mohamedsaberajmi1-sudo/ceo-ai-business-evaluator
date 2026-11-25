"""Localization strings for the chatbot, reports and Streamlit UI.

English, French and Tunisian Arabic (Arabic script) are supported. The app also
accepts Tunisian Arabic written in Latin script in the conversation parser.
"""

from __future__ import annotations

from .utils import normalize_language

# key: {language: text}
TRANSLATIONS: dict[str, dict[str, str]] = {
    "tagline": {
        "en": "Business idea evaluation & performance analytics",
        "fr": "Évaluation d'idées et analyse de performance",
        "ar": "تقييم أفكار المشاريع وتحليل الأداء",
    },
    "mode": {"en": "Mode", "fr": "Mode", "ar": "الوضع"},
    "mode_idea": {"en": "Evaluate an idea", "fr": "Évaluer une idée", "ar": "تقييم فكرة"},
    "mode_business": {"en": "Analyze a business", "fr": "Analyser une entreprise", "ar": "تحليل مشروع قائم"},
    "mode_history": {"en": "History & compare", "fr": "Historique & comparaison", "ar": "السجل والمقارنة"},
    "language": {"en": "Language", "fr": "Langue", "ar": "اللغة"},
    "market_preset": {"en": "Market preset", "fr": "Marché prédéfini", "ar": "سوق افتراضي"},
    "reset_chat": {"en": "New conversation", "fr": "Nouvelle conversation", "ar": "محادثة جديدة"},
    "export_pdf": {"en": "Export report to PDF", "fr": "Exporter le rapport en PDF", "ar": "تصدير التقرير PDF"},
    "download_pdf": {"en": "Download PDF", "fr": "Télécharger le PDF", "ar": "تحميل PDF"},
    "chat_placeholder": {
        "en": "Describe your idea or answer the question…",
        "fr": "Décrivez votre idée ou répondez à la question…",
        "ar": "اوصف فكرتك أو أجب على السؤال…",
    },
    "llm_offline": {
        "en": "Offline rule-based explanations",
        "fr": "Explications hors ligne (règles)",
        "ar": "شرح دون اتصال بالقواعد",
    },
    "llm_online": {"en": "LLM connected", "fr": "LLM connecté", "ar": "الذكاء الاصطناعي متصل"},
    "llm_error": {"en": "LLM unavailable — fallback used", "fr": "LLM indisponible — repli utilisé", "ar": "الذكاء الاصطناعي غير متاح — تم استعمال البديل"},
    "reset_done": {
        "en": "Starting a new conversation.",
        "fr": "Nouvelle conversation démarrée.",
        "ar": "بدأنا محادثة جديدة.",
    },
    "progress": {"en": "Information collected", "fr": "Informations collectées", "ar": "المعلومات المجمعة"},
    "evaluation_now": {
        "en": "You can type “evaluate now” at any time to use sensible defaults for missing answers.",
        "fr": "Vous pouvez écrire « évaluer maintenant » pour utiliser des valeurs par défaut.",
        "ar": "يمكنك كتابة «قيّم الآن» لاستعمال قيم افتراضية للمعلومات الناقصة.",
    },
    "skip_hint": {
        "en": "Type “skip” to use a default value.",
        "fr": "Écrivez « passer » pour utiliser une valeur par défaut.",
        "ar": "اكتب «تجاوز» لاستعمال قيمة افتراضية.",
    },
    "invalid_input": {
        "en": "I could not interpret that answer, so I used a reasonable default. You can correct it later by starting a new conversation.",
        "fr": "Je n'ai pas pu interpréter cette réponse ; une valeur par défaut a été utilisée. Vous pourrez la corriger dans une nouvelle conversation.",
        "ar": "لم أتمكن من فهم الجواب، استعملت قيمة افتراضية. يمكنك تصحيحها في محادثة جديدة.",
    },
    "report_ready": {
        "en": "Your report is ready.",
        "fr": "Votre rapport est prêt.",
        "ar": "تقريرك جاهز.",
    },
    "report_intro": {
        "en": "Here is the structured evaluation. The ML probability is an estimate learned from historical patterns, not a guarantee.",
        "fr": "Voici l'évaluation structurée. La probabilité ML est une estimation apprise sur des motifs historiques, pas une garantie.",
        "ar": "هذا هو التقييم المنظم. احتمال النجاح من نموذج تعلم آلي تقديري وليس ضمانًا.",
    },
    "overview": {"en": "Overview", "fr": "Vue d'ensemble", "ar": "نظرة عامة"},
    "scores": {"en": "Sub-scores", "fr": "Sous-scores", "ar": "الدرجات الفرعية"},
    "financials": {"en": "Financials", "fr": "Finances", "ar": "المالية"},
    "swot": {"en": "SWOT", "fr": "SWOT", "ar": "تحليل SWOT"},
    "risks_actions": {"en": "Risks & action plan", "fr": "Risques & plan d'action", "ar": "المخاطر وخطة العمل"},
    "what_if": {"en": "What-if simulator", "fr": "Simulateur what-if", "ar": "محاكي ماذا لو"},
    "forecast": {"en": "Forecast", "fr": "Prévisions", "ar": "التوقعات"},
    "benchmarks": {"en": "Benchmarks", "fr": "Références sectorielles", "ar": "المعايير القطاعية"},
    "kpis": {"en": "KPIs", "fr": "Indicateurs clés", "ar": "مؤشرات الأداء"},
    "overall_score": {"en": "Overall viability", "fr": "Viabilité globale", "ar": "الجدوى الإجمالية"},
    "success_probability": {"en": "Success probability", "fr": "Probabilité de succès", "ar": "احتمال النجاح"},
    "confidence": {"en": "Confidence", "fr": "Confiance", "ar": "الثقة"},
    "verdict": {"en": "Verdict", "fr": "Verdict", "ar": "الحكم"},
    "roi": {"en": "Estimated ROI", "fr": "ROI estimé", "ar": "العائد المتوقع"},
    "breakeven": {"en": "Break-even", "fr": "Seuil de rentabilité", "ar": "نقطة التعادل"},
    "profit_range": {"en": "Monthly profit scenarios", "fr": "Scénarios de profit mensuel", "ar": "سيناريوهات الربح الشهري"},
    "pessimistic": {"en": "Pessimistic", "fr": "Pessimiste", "ar": "متشائم"},
    "realistic": {"en": "Realistic", "fr": "Réaliste", "ar": "واقعي"},
    "optimistic": {"en": "Optimistic", "fr": "Optimiste", "ar": "متفائل"},
    "months": {"en": "months", "fr": "mois", "ar": "أشهر"},
    "month": {"en": "Month", "fr": "Mois", "ar": "الشهر"},
    "score_out_of": {"en": "{score}/10", "fr": "{score}/10", "ar": "{score}/10"},
    "top_drivers": {"en": "What drives the score", "fr": "Ce qui influence le score", "ar": "ما يؤثر على الدرجة"},
    "positive_drivers": {"en": "Positive drivers", "fr": "Facteurs positifs", "ar": "العوامل الإيجابية"},
    "negative_drivers": {"en": "Negative drivers", "fr": "Facteurs négatifs", "ar": "العوامل السلبية"},
    "explanation_note": {
        "en": "Drivers are estimated with SHAP when available, otherwise with model feature importance.",
        "fr": "Les facteurs sont estimés avec SHAP si disponible, sinon avec l'importance des variables du modèle.",
        "ar": "تُقدَّر العوامل بواسطة SHAP عند توفره، وإلا عبر أهمية المتغيرات في النموذج.",
    },
    "strengths": {"en": "Strengths", "fr": "Forces", "ar": "نقاط القوة"},
    "weaknesses": {"en": "Weaknesses", "fr": "Faiblesses", "ar": "نقاط الضعف"},
    "opportunities": {"en": "Opportunities", "fr": "Opportunités", "ar": "الفرص"},
    "threats": {"en": "Threats", "fr": "Menaces", "ar": "التهديدات"},
    "key_risks": {"en": "Key risks", "fr": "Risques principaux", "ar": "المخاطر الرئيسية"},
    "recommended_modifications": {
        "en": "Recommended modifications",
        "fr": "Modifications recommandées",
        "ar": "التعديلات الموصى بها",
    },
    "score_gain": {"en": "score gain", "fr": "gain de score", "ar": "زيادة الدرجة"},
    "roi_gain": {"en": "ROI gain", "fr": "gain de ROI", "ar": "زيادة العائد"},
    "priority": {"en": "Priority", "fr": "Priorité", "ar": "الأولوية"},
    "impact": {"en": "Estimated impact", "fr": "Impact estimé", "ar": "الأثر المقدر"},
    "target_segment": {"en": "Recommended target segment", "fr": "Segment cible recommandé", "ar": "القطاع المستهدف المقترح"},
    "go_to_market": {"en": "Go-to-market suggestions", "fr": "Suggestions de mise sur le marché", "ar": "اقتراحات الدخول للسوق"},
    "what_if_help": {
        "en": "Move the assumptions to see their live effect on the score, success probability and ROI.",
        "fr": "Modifiez les hypothèses pour voir leur effet sur le score, la probabilité de succès et le ROI.",
        "ar": "غيّر الافتراضات لرؤية تأثيرها الفوري على الدرجة واحتمال النجاح والعائد.",
    },
    "base_case": {"en": "Base case", "fr": "Scénario de base", "ar": "الحالة الأساسية"},
    "modified_case": {"en": "Modified case", "fr": "Scénario modifié", "ar": "الحالة المعدلة"},
    "price": {"en": "Average price", "fr": "Prix moyen", "ar": "السعر المتوسط"},
    "budget": {"en": "Initial budget", "fr": "Budget initial", "ar": "رأس المال الأولي"},
    "marketing": {"en": "Monthly marketing", "fr": "Marketing mensuel", "ar": "التسويق الشهري"},
    "expected_customers": {"en": "Customers / month", "fr": "Clients / mois", "ar": "العملاء / شهر"},
    "differentiation": {"en": "Differentiation", "fr": "Différenciation", "ar": "التميّز"},
    "forecast_next": {"en": "Next {periods} months", "fr": "{periods} prochains mois", "ar": "الأشهر {periods} القادمة"},
    "forecast_method": {"en": "Method", "fr": "Méthode", "ar": "الطريقة"},
    "confidence_band": {"en": "Confidence band", "fr": "Intervalle de confiance", "ar": "مجال الثقة"},
    "revenue": {"en": "Revenue", "fr": "Chiffre d'affaires", "ar": "الإيرادات"},
    "costs": {"en": "Costs", "fr": "Coûts", "ar": "التكاليف"},
    "profit": {"en": "Profit", "fr": "Profit", "ar": "الربح"},
    "margin": {"en": "Net margin", "fr": "Marge nette", "ar": "الهامش الصافي"},
    "growth": {"en": "Growth", "fr": "Croissance", "ar": "النمو"},
    "churn": {"en": "Churn", "fr": "Attrition", "ar": "تسرب العملاء"},
    "cac": {"en": "CAC", "fr": "CAC", "ar": "تكلفة اكتساب العميل"},
    "ltv": {"en": "LTV", "fr": "LTV", "ar": "القيمة الدائمة للعميل"},
    "ltv_cac": {"en": "LTV / CAC", "fr": "LTV / CAC", "ar": "LTV / CAC"},
    "marketing_ratio": {"en": "Marketing / revenue", "fr": "Marketing / CA", "ar": "التسويق / الإيرادات"},
    "avg_order_value": {"en": "Avg. revenue / customer", "fr": "CA moyen / client", "ar": "متوسط الإيراد لكل عميل"},
    "customers": {"en": "Customers", "fr": "Clients", "ar": "العملاء"},
    "benchmark": {"en": "Benchmark", "fr": "Référence", "ar": "المعيار"},
    "actual": {"en": "Actual", "fr": "Réel", "ar": "الفعلي"},
    "gap": {"en": "Gap", "fr": "Écart", "ar": "الفارق"},
    "status": {"en": "Status", "fr": "Statut", "ar": "الحالة"},
    "good": {"en": "Good", "fr": "Bon", "ar": "جيد"},
    "watch": {"en": "Watch", "fr": "À surveiller", "ar": "للمراقبة"},
    "bad": {"en": "Needs action", "fr": "Action requise", "ar": "يتطلب إجراء"},
    "detected_problems": {"en": "Detected problems", "fr": "Problèmes détectés", "ar": "المشاكل المكتشفة"},
    "prioritized_actions": {"en": "Prioritized actions", "fr": "Actions prioritaires", "ar": "الإجراءات حسب الأولوية"},
    "no_problems": {
        "en": "No major red flag was detected. Keep monitoring the KPIs.",
        "fr": "Aucun signal majeur détecté. Continuez à suivre les indicateurs.",
        "ar": "لم يتم اكتشاف إشارة خطر كبيرة. واصل متابعة المؤشرات.",
    },
    "estimated_impact": {"en": "Estimated impact", "fr": "Impact estimé", "ar": "الأثر المقدر"},
    "rationale": {"en": "Why it matters", "fr": "Pourquoi c'est important", "ar": "لماذا يهم"},
    "upload_data": {"en": "Upload sales data (CSV / Excel)", "fr": "Importer des données (CSV / Excel)", "ar": "ارفع بيانات المبيعات (CSV / Excel)"},
    "upload_help": {
        "en": "Columns: date/month, revenue, cost, customers, new_customers, marketing_spend, price. Aliases are accepted.",
        "fr": "Colonnes : date/mois, revenue, cost, customers, new_customers, marketing_spend, price. Des alias sont acceptés.",
        "ar": "الأعمدة: date/month، revenue، cost، customers، new_customers، marketing_spend، price. تُقبل أسماء بديلة.",
    },
    "sector": {"en": "Sector", "fr": "Secteur", "ar": "القطاع"},
    "analyze": {"en": "Analyze uploaded data", "fr": "Analyser les données", "ar": "تحليل البيانات"},
    "manual_snapshot": {"en": "Or use a manual monthly snapshot", "fr": "Ou utilisez un instantané mensuel", "ar": "أو استعمل لقطة شهرية يدوية"},
    "monthly_revenue": {"en": "Monthly revenue", "fr": "Chiffre d'affaires mensuel", "ar": "الإيرادات الشهرية"},
    "monthly_cost": {"en": "Monthly total costs", "fr": "Coûts mensuels totaux", "ar": "التكاليف الشهرية الإجمالية"},
    "monthly_marketing": {"en": "Monthly marketing spend", "fr": "Dépenses marketing mensuelles", "ar": "مصروف التسويق الشهري"},
    "monthly_growth_pct": {"en": "Assumed monthly growth (%)", "fr": "Croissance mensuelle supposée (%)", "ar": "النمو الشهري المفترض (%)"},
    "analyze_manual": {"en": "Analyze snapshot", "fr": "Analyser l'instantané", "ar": "تحليل اللقطة"},
    "file_required": {"en": "Please upload a file first.", "fr": "Veuillez d'abord importer un fichier.", "ar": "يرجى رفع ملف أولاً."},
    "analysis_complete": {"en": "Business analysis complete.", "fr": "Analyse de l'entreprise terminée.", "ar": "اكتمل تحليل المشروع."},
    "data_preview": {"en": "Data preview", "fr": "Aperçu des données", "ar": "معاينة البيانات"},
    "data_quality": {"en": "Data quality", "fr": "Qualité des données", "ar": "جودة البيانات"},
    "history_title": {"en": "Saved evaluations", "fr": "Évaluations sauvegardées", "ar": "التقييمات المحفوظة"},
    "no_history": {
        "en": "No saved evaluation yet. Complete an idea evaluation first.",
        "fr": "Aucune évaluation sauvegardée. Terminez d'abord une évaluation.",
        "ar": "لا يوجد تقييم محفوظ بعد. أكمل تقييم فكرة أولاً.",
    },
    "select_evaluation": {"en": "Select an evaluation", "fr": "Choisir une évaluation", "ar": "اختر تقييمًا"},
    "load_evaluation": {"en": "Load", "fr": "Charger", "ar": "تحميل"},
    "compare_title": {"en": "Compare ideas", "fr": "Comparer des idées", "ar": "مقارنة الأفكار"},
    "select_2_3": {
        "en": "Select 2 or 3 saved evaluations",
        "fr": "Sélectionnez 2 ou 3 évaluations",
        "ar": "اختر تقييمين أو ثلاثة",
    },
    "compare": {"en": "Compare", "fr": "Comparer", "ar": "قارن"},
    "saved_at": {"en": "Saved", "fr": "Sauvegardé", "ar": "تم الحفظ"},
    "idea": {"en": "Idea", "fr": "Idée", "ar": "فكرة"},
    "business": {"en": "Business", "fr": "Entreprise", "ar": "مشروع"},
    "disclaimer_title": {"en": "Important", "fr": "Important", "ar": "مهم"},
    "disclaimer": {
        "en": "ROI, profit and break-even are estimates based on synthetic/uploaded data and assumptions. They are not financial advice.",
        "fr": "Le ROI, le profit et le seuil de rentabilité sont des estimations basées sur des données synthétiques/importées et des hypothèses. Ce n'est pas un conseil financier.",
        "ar": "العائد والربح ونقطة التعادل تقديرات مبنية على بيانات اصطناعية/مرفوعة وافتراضات. هذا ليس نصيحة مالية.",
    },
    "agent_greeting": {
        "en": "Hello, I'm {bot}. I evaluate business ideas and existing businesses with real ML models. Let's start: describe the business idea you want to launch.",
        "fr": "Bonjour, je suis {bot}. J'évalue les idées et les entreprises avec de vrais modèles ML. Commençons : décrivez l'idée que vous voulez lancer.",
        "ar": "مرحبًا، أنا {bot}. نقيّم أفكار المشاريع والمشاريع القائمة بنماذج تعلم آلي حقيقية. لنبدأ: اوصف فكرة المشروع التي تريد إطلاقها.",
    },
    "greeting_short": {
        "en": "New conversation started. I'm {bot}.",
        "fr": "Nouvelle conversation démarrée. Je suis {bot}.",
        "ar": "بدأت محادثة جديدة. أنا {bot}.",
    },
    "ask_description": {
        "en": "Describe your idea in a few sentences (problem, offer, and why customers would buy it).",
        "fr": "Décrivez votre idée en quelques phrases (problème, offre, et pourquoi les clients achèteraient).",
        "ar": "اوصف فكرتك في بضعة أسطر (المشكلة، العرض، ولماذا سيشتري العملاء).",
    },
    "ask_sector": {
        "en": "Which sector is closest? {options}",
        "fr": "Quel secteur est le plus proche ? {options}",
        "ar": "ما هو أقرب قطاع؟ {options}",
    },
    "ask_target": {
        "en": "Who are your target customers? You can say B2C, B2B, marketplace, and describe the segment.",
        "fr": "Qui sont vos clients cibles ? Indiquez B2C, B2B, marketplace et décrivez le segment.",
        "ar": "من هم عملاؤك المستهدفون؟ يمكنك ذكر B2C أو B2B أو سوق إلكتروني ووصف الشريحة.",
    },
    "ask_location": {
        "en": "In which country and city will you operate?",
        "fr": "Dans quel pays et quelle ville allez-vous opérer ?",
        "ar": "في أي بلد ومدينة ستشتغل؟",
    },
    "ask_budget": {
        "en": "What is your initial budget? (For example 30000)",
        "fr": "Quel est votre budget initial ? (Par exemple 30000)",
        "ar": "ما هو رأس المال الأولي؟ (مثلاً 30000)",
    },
    "ask_team": {
        "en": "How many people will work in the business at launch?",
        "fr": "Combien de personnes travailleront dans l'entreprise au lancement ?",
        "ar": "قداش من شخص سيعمل في المشروع عند الإطلاق؟",
    },
    "ask_price": {
        "en": "What is your average price per sale or subscription per month?",
        "fr": "Quel est votre prix moyen par vente ou abonnement mensuel ?",
        "ar": "ما هو متوسط السعر لكل عملية بيع أو اشتراك شهري؟",
    },
    "ask_customers": {
        "en": "How many paying customers (or orders) do you expect per month at launch?",
        "fr": "Combien de clients payants (ou commandes) attendez-vous par mois au lancement ?",
        "ar": "قداش من عميل يدفع (أو طلبية) تتوقع في الشهر عند الإطلاق؟",
    },
    "ask_revenue_model": {
        "en": "What is the revenue model? subscription, one_time, commission, freemium, advertising or hybrid.",
        "fr": "Quel est le modèle de revenus ? subscription, one_time, commission, freemium, advertising ou hybrid.",
        "ar": "ما هو نموذج الإيرادات؟ اشتراك، دفع مرة واحدة، عمولة، مجاني جزئي، إعلانات أو مختلط.",
    },
    "ask_competition": {
        "en": "How intense is competition? Give a number from 1 (low) to 10 (very high).",
        "fr": "Quelle est l'intensité de la concurrence ? Donnez un nombre de 1 (faible) à 10 (très forte).",
        "ar": "قداش المنافسة قوية؟ أعطِ رقمًا من 1 (ضعيفة) إلى 10 (قوية جدًا).",
    },
    "ask_differentiation": {
        "en": "How differentiated is your offer? 1 (copycat) to 10 (clearly unique).",
        "fr": "Quel est le niveau de différenciation ? 1 (copie) à 10 (clairement unique).",
        "ar": "قداش عرضك متميز؟ من 1 (تقليد) إلى 10 (متميز بوضوح).",
    },
    "ask_timeline": {
        "en": "How many months do you need before launching?",
        "fr": "Combien de mois vous faut-il avant le lancement ?",
        "ar": "قداش من شهر تحتاج قبل الإطلاق؟",
    },
    "received": {"en": "Got it.", "fr": "Compris.", "ar": "فهمت."},
    "default_used": {
        "en": "I used a default value: {value}.",
        "fr": "J'ai utilisé une valeur par défaut : {value}.",
        "ar": "استعملت قيمة افتراضية: {value}.",
    },
    "evaluation_started": {
        "en": "I have enough information. I'm running the ML models and scoring engine now…",
        "fr": "J'ai assez d'informations. Je lance les modèles ML et le moteur de score…",
        "ar": "عندي معلومات كافية. نبدأ تشغيل نماذج التعلم الآلي ومحرك التقييم…",
    },
    "business_greeting": {
        "en": "Upload a monthly CSV/Excel file in the sidebar, or use the manual monthly snapshot. I will compute KPIs, detect problems, compare with sector benchmarks and forecast revenue/profit.",
        "fr": "Importez un fichier mensuel CSV/Excel dans la barre latérale, ou utilisez l'instantané manuel. Je calculerai les KPI, détecterai les problèmes, comparerai aux références et prévoirai le CA/profit.",
        "ar": "ارفع ملف CSV/Excel شهري في الشريط الجانبي أو استعمل اللقطة اليدوية. سأحسب المؤشرات وأكتشف المشاكل وأقارن بالمعايير وأتوقع الإيرادات والأرباح.",
    },
    "history_help": {
        "en": "Select a saved evaluation to reload its report, or compare 2–3 ideas side by side.",
        "fr": "Sélectionnez une évaluation sauvegardée pour recharger son rapport, ou comparez 2–3 idées côte à côte.",
        "ar": "اختر تقييمًا محفوظًا لتحميل تقريره، أو قارن 2–3 أفكار جنبًا إلى جنب.",
    },
    "ask_anything": {
        "en": "You can now ask me: why this score, what is the ROI, what are the main risks, or how to improve.",
        "fr": "Vous pouvez maintenant me demander : pourquoi ce score, quel est le ROI, quels sont les risques, ou comment améliorer.",
        "ar": "يمكنك الآن سؤالي: لماذا هذه الدرجة، ما هو العائد، ما هي المخاطر، أو كيف نحسّن.",
    },
    "answer_why": {
        "en": "The overall score is {score}/100. The ML model estimates a {prob}% success probability. Main positive drivers: {positives}. Main negative drivers: {negatives}.",
        "fr": "Le score global est {score}/100. Le modèle ML estime une probabilité de succès de {prob}%. Facteurs positifs : {positives}. Facteurs négatifs : {negatives}.",
        "ar": "الدرجة الإجمالية {score}/100. النموذج يقدّر احتمال النجاح بـ {prob}%. العوامل الإيجابية: {positives}. العوامل السلبية: {negatives}.",
    },
    "answer_roi": {
        "en": "Estimated ROI is {roi}% with break-even around {months} months. The realistic monthly profit is about {profit}; the range is {low} to {high}. This is an estimate, not financial advice.",
        "fr": "Le ROI estimé est {roi}% avec un seuil de rentabilité autour de {months} mois. Le profit mensuel réaliste est d'environ {profit}, avec une fourchette de {low} à {high}. C'est une estimation, pas un conseil financier.",
        "ar": "العائد المتوقع {roi}% ونقطة التعادل حوالي {months} شهرًا. الربح الشهري الواقعي حوالي {profit}، والمجال من {low} إلى {high}. هذا تقدير وليس نصيحة مالية.",
    },
    "answer_risks": {
        "en": "The main risks are: {risks}. The most useful mitigations are: {actions}.",
        "fr": "Les principaux risques sont : {risks}. Les mesures les plus utiles sont : {actions}.",
        "ar": "المخاطر الرئيسية: {risks}. أفضل الإجراءات للتخفيف: {actions}.",
    },
    "answer_improve": {
        "en": "The fastest improvements are: {actions}. The best action could add about {points} points to the score and {roi_points} ROI points.",
        "fr": "Les améliorations les plus rapides sont : {actions}. La meilleure action pourrait ajouter environ {points} points au score et {roi_points} points de ROI.",
        "ar": "أسرع التحسينات: {actions}. أفضل إجراء قد يضيف حوالي {points} نقطة للدرجة و{roi_points} نقطة للعائد.",
    },
    "answer_summary": {
        "en": "Verdict: {verdict}. Score {score}/100, success probability {prob}%, ROI {roi}%, break-even {months} months.",
        "fr": "Verdict : {verdict}. Score {score}/100, probabilité de succès {prob}%, ROI {roi}%, seuil de rentabilité {months} mois.",
        "ar": "الحكم: {verdict}. الدرجة {score}/100، احتمال النجاح {prob}%، العائد {roi}%، نقطة التعادل {months} شهرًا.",
    },
    "answer_default": {
        "en": "I can explain the score, the ROI, the risks or the recommended improvements. You can also type “new idea” to restart.",
        "fr": "Je peux expliquer le score, le ROI, les risques ou les améliorations recommandées. Écrivez « nouvelle idée » pour recommencer.",
        "ar": "أستطيع شرح الدرجة أو العائد أو المخاطر أو التحسينات المقترحة. اكتب «فكرة جديدة» لإعادة البدء.",
    },
    "verdict_succeed": {
        "en": "Likely to succeed if execution is strong",
        "fr": "Probablement viable si l'exécution est solide",
        "ar": "مرجّح أن ينجح إذا كان التنفيذ قويًا",
    },
    "verdict_risky": {
        "en": "Risky — needs validation and adjustments",
        "fr": "Risqué — nécessite validation et ajustements",
        "ar": "محفوف بالمخاطر — يحتاج تحققًا وتعديلات",
    },
    "verdict_unlikely": {
        "en": "Unlikely to succeed in its current form",
        "fr": "Peu probable dans sa forme actuelle",
        "ar": "غير مرجّح في شكله الحالي",
    },
    "not_financial_advice": {
        "en": "Estimate only — not financial advice.",
        "fr": "Estimation uniquement — pas un conseil financier.",
        "ar": "تقدير فقط — ليس نصيحة مالية.",
    },
    "error": {"en": "An error occurred", "fr": "Une erreur est survenue", "ar": "حدث خطأ"},
    "model_missing": {
        "en": "Trained models were not found. Run `python -m src.train`; a heuristic fallback is being used.",
        "fr": "Les modèles entraînés sont absents. Exécutez `python -m src.train` ; un repli heuristique est utilisé.",
        "ar": "النماذج المدربة غير موجودة. شغّل `python -m src.train`؛ يتم استعمال تقدير احتياطي.",
    },
}


TRANSLATIONS.update(
    {
        "crit_market_size": {"en": "Market size & demand", "fr": "Taille du marché & demande", "ar": "حجم السوق والطلب"},
        "crit_market_growth": {"en": "Market growth / trend", "fr": "Croissance / tendance du marché", "ar": "نمو واتجاه السوق"},
        "crit_audience_fit": {"en": "Target audience fit", "fr": "Adéquation à la cible", "ar": "ملاءمة الجمهور المستهدف"},
        "crit_competition": {"en": "Competitive position", "fr": "Position concurrentielle", "ar": "الموقع التنافسي"},
        "crit_differentiation": {"en": "Differentiation", "fr": "Différenciation", "ar": "التميّز"},
        "crit_startup_cost": {"en": "Startup cost & capital", "fr": "Coût de lancement & capital", "ar": "تكلفة الانطلاق ورأس المال"},
        "crit_operations": {"en": "Operational feasibility", "fr": "Faisabilité opérationnelle", "ar": "الجدوى التشغيلية"},
        "crit_revenue_model": {"en": "Revenue model strength", "fr": "Solidité du modèle de revenus", "ar": "قوة نموذج الإيرادات"},
        "crit_scalability": {"en": "Scalability", "fr": "Scalabilité", "ar": "قابلية التوسع"},
        "crit_timing": {"en": "Timing", "fr": "Timing", "ar": "التوقيت"},
        "crit_regulation": {"en": "Regulatory risk", "fr": "Risque réglementaire", "ar": "المخاطر التنظيمية"},
        "crit_team": {"en": "Team readiness", "fr": "Préparation de l'équipe", "ar": "جاهزية الفريق"},
        "swot_strength_line": {
            "en": "{criterion} is a relative strength ({score}/10).",
            "fr": "{criterion} est une force relative ({score}/10).",
            "ar": "{criterion} نقطة قوة نسبية ({score}/10).",
        },
        "swot_weakness_line": {
            "en": "{criterion} is below the desired level ({score}/10).",
            "fr": "{criterion} est sous le niveau souhaité ({score}/10).",
            "ar": "{criterion} أقل من المستوى المطلوب ({score}/10).",
        },
        "swot_strength_generic": {
            "en": "The idea has credible building blocks but needs validation.",
            "fr": "L'idée a des bases crédibles mais doit être validée.",
            "ar": "للفكرة أساسيات معقولة لكنها تحتاج تحققًا.",
        },
        "swot_weakness_generic": {
            "en": "Several assumptions remain unproven.",
            "fr": "Plusieurs hypothèses restent à prouver.",
            "ar": "عدة افتراضات ما زالت غير مثبتة.",
        },
        "swot_opportunity_growth": {
            "en": "The market trend is favorable ({score}/10), leaving room for demand growth.",
            "fr": "La tendance du marché est favorable ({score}/10), avec un potentiel de demande.",
            "ar": "اتجاه السوق إيجابي ({score}/10) مع مجال لنمو الطلب.",
        },
        "swot_opportunity_timing": {
            "en": "Timing looks acceptable for a focused MVP launch.",
            "fr": "Le timing semble acceptable pour un MVP ciblé.",
            "ar": "التوقيت يبدو مناسبًا لإطلاق MVP مركز.",
        },
        "swot_opportunity_segment": {
            "en": "A focused {market} segment can be targeted first.",
            "fr": "Un segment {market} ciblé peut être attaqué en premier.",
            "ar": "يمكن استهداف شريحة {market} محددة أولاً.",
        },
        "swot_opportunity_profit": {
            "en": "The realistic scenario is profitable under current assumptions.",
            "fr": "Le scénario réaliste est rentable avec les hypothèses actuelles.",
            "ar": "السيناريو الواقعي مربح في ظل الافتراضات الحالية.",
        },
        "swot_opportunity_generic": {
            "en": "Validate demand with a low-cost pilot before scaling.",
            "fr": "Validez la demande avec un pilote peu coûteux avant de scaler.",
            "ar": "تحقق من الطلب عبر تجربة منخفضة التكلفة قبل التوسع.",
        },
        "swot_threat_competition": {
            "en": "Competition is intense ({score}/10); a clear niche is needed.",
            "fr": "La concurrence est intense ({score}/10) ; une niche claire est nécessaire.",
            "ar": "المنافسة شديدة ({score}/10) وتتطلب تخصصًا واضحًا.",
        },
        "swot_threat_regulation": {
            "en": "Regulatory work may delay launch and add cost.",
            "fr": "Les démarches réglementaires peuvent retarder le lancement et coûter cher.",
            "ar": "الإجراءات التنظيمية قد تؤخر الإطلاق وتزيد التكلفة.",
        },
        "swot_threat_capital": {
            "en": "The capital base looks thin for this sector.",
            "fr": "Le capital semble faible pour ce secteur.",
            "ar": "رأس المال يبدو ضعيفًا لهذا القطاع.",
        },
        "swot_threat_operations": {
            "en": "Operational complexity could hurt execution.",
            "fr": "La complexité opérationnelle peut nuire à l'exécution.",
            "ar": "التعقيد التشغيلي قد يضر بالتنفيذ.",
        },
        "swot_threat_team": {
            "en": "The team may lack a critical skill.",
            "fr": "L'équipe peut manquer d'une compétence critique.",
            "ar": "قد يفتقد الفريق مهارة أساسية.",
        },
        "swot_threat_generic": {
            "en": "Execution and demand assumptions must be validated.",
            "fr": "Les hypothèses d'exécution et de demande doivent être validées.",
            "ar": "يجب التحقق من افتراضات التنفيذ والطلب.",
        },
        "risk_competition": {
            "en": "High competitive intensity can increase acquisition costs.",
            "fr": "Une concurrence forte peut augmenter les coûts d'acquisition.",
            "ar": "شدة المنافسة قد ترفع تكاليف اكتساب العملاء.",
        },
        "risk_regulation": {
            "en": "Regulatory approvals or compliance costs may delay launch.",
            "fr": "Les autorisations ou coûts de conformité peuvent retarder le lancement.",
            "ar": "التراخيص أو تكاليف الامتثال قد تؤخر الإطلاق.",
        },
        "risk_capital": {
            "en": "The required capital may exceed the available budget.",
            "fr": "Le capital requis peut dépasser le budget disponible.",
            "ar": "قد يتجاوز رأس المال المطلوب الميزانية المتاحة.",
        },
        "risk_operations": {
            "en": "Operational complexity may strain a small team.",
            "fr": "La complexité opérationnelle peut épuiser une petite équipe.",
            "ar": "التعقيد التشغيلي قد يرهق فريقًا صغيرًا.",
        },
        "risk_team": {
            "en": "The team readiness and critical skills are not yet proven.",
            "fr": "La préparation de l'équipe et les compétences clés ne sont pas prouvées.",
            "ar": "جاهزية الفريق والمهارات الأساسية غير مثبتة بعد.",
        },
        "risk_revenue": {
            "en": "The revenue model may not convert demand into recurring cash.",
            "fr": "Le modèle de revenus peut ne pas transformer la demande en cash récurrent.",
            "ar": "قد لا يحول نموذج الإيرادات الطلب إلى تدفق نقدي متكرر.",
        },
        "risk_timing": {
            "en": "A long or poorly sequenced launch can exhaust resources.",
            "fr": "Un lancement long ou mal séquencé peut épuiser les ressources.",
            "ar": "إطلاق طويل أو غير منظم قد يستنزف الموارد.",
        },
        "risk_growth": {
            "en": "Slow market growth could cap demand.",
            "fr": "Une croissance lente du marché peut plafonner la demande.",
            "ar": "نمو السوق البطيء قد يحد من الطلب.",
        },
        "risk_differentiation": {
            "en": "Weak differentiation may make the offer easy to copy.",
            "fr": "Une différenciation faible rend l'offre facile à copier.",
            "ar": "ضعف التميّز قد يجعل العرض سهل التقليد.",
        },
        "risk_generic": {
            "en": "Demand, pricing and execution must be tested before scaling.",
            "fr": "La demande, le prix et l'exécution doivent être testés avant de scaler.",
            "ar": "يجب اختبار الطلب والتسعير والتنفيذ قبل التوسع.",
        },
        "action_differentiation": {
            "en": "Sharpen the positioning with one unique promise and proof points",
            "fr": "Affiner le positionnement avec une promesse unique et des preuves",
            "ar": "توضيح التموقع بوعد فريد وأدلة ملموسة",
        },
        "action_competition": {
            "en": "Target a narrower underserved niche to reduce direct competition",
            "fr": "Cibler une niche plus étroite et mal servie pour réduire la concurrence",
            "ar": "استهداف شريحة أضيق ومهملة لتقليل المنافسة المباشرة",
        },
        "action_budget": {
            "en": "Secure about 30% more capital or reduce the launch scope",
            "fr": "Sécuriser environ 30% de capital en plus ou réduire le périmètre",
            "ar": "تأمين حوالي 30% إضافية من رأس المال أو تقليص نطاق الإطلاق",
        },
        "action_team": {
            "en": "Add one key hire/co-founder and a 90-day execution plan",
            "fr": "Ajouter un recrutement/co-fondateur clé et un plan d'exécution à 90 jours",
            "ar": "إضافة كفاءة أساسية أو شريك وتنفيذ خطة 90 يومًا",
        },
        "action_revenue": {
            "en": "Test a subscription/hybrid pricing model with a 8% price increase",
            "fr": "Tester un modèle d'abonnement/hybride avec un prix +8%",
            "ar": "اختبار تسعير اشتراك/مختلط مع رفع السعر 8%",
        },
        "action_regulation": {
            "en": "Budget legal/compliance work early and document certifications",
            "fr": "Budgéter la conformité tôt et documenter les certifications",
            "ar": "تخصيص ميزانية للامتثال مبكرًا وتوثيق الشهادات",
        },
        "action_growth": {
            "en": "Test the adjacent customer segment with the strongest growth",
            "fr": "Tester le segment adjacent à la plus forte croissance",
            "ar": "اختبار الشريحة المجاورة ذات النمو الأقوى",
        },
        "action_operations": {
            "en": "Simplify to one core product/process and partner for the rest",
            "fr": "Simplifier à un produit/processus central et externaliser le reste",
            "ar": "تبسيط العرض لمنتج أو عملية أساسية والشراكة في الباقي",
        },
        "action_timing": {
            "en": "Launch a smaller MVP sooner to shorten time-to-market",
            "fr": "Lancer un MVP plus petit plus tôt pour réduire le time-to-market",
            "ar": "إطلاق MVP أصغر وأسرع لتقصير وقت الوصول للسوق",
        },
        "action_audience": {
            "en": "Run 20 customer interviews to validate the chosen segment",
            "fr": "Mener 20 entretiens clients pour valider le segment choisi",
            "ar": "إجراء 20 مقابلة مع العملاء للتحقق من الشريحة المختارة",
        },
        "cost_note_differentiation": {
            "en": "Low cost: positioning and proof assets",
            "fr": "Coût faible : positionnement et preuves",
            "ar": "تكلفة منخفضة: التموقع والأدلة",
        },
        "cost_note_competition": {
            "en": "Low cost: segment focus",
            "fr": "Coût faible : recentrage sur un segment",
            "ar": "تكلفة منخفضة: تركيز على شريحة",
        },
        "cost_note_capital": {
            "en": "Fundraising or scope reduction",
            "fr": "Levée de fonds ou réduction du périmètre",
            "ar": "تمويل إضافي أو تقليص النطاق",
        },
        "cost_note_team": {
            "en": "Salary/equity cost",
            "fr": "Coût salarial/équity",
            "ar": "تكلفة راتب أو حصة",
        },
        "cost_note_revenue": {
            "en": "Low cost, needs pricing tests",
            "fr": "Coût faible, nécessite des tests de prix",
            "ar": "تكلفة منخفضة مع اختبارات تسعير",
        },
        "cost_note_regulation": {
            "en": "Legal and compliance budget",
            "fr": "Budget juridique et conformité",
            "ar": "ميزانية قانونية وامتثال",
        },
        "cost_note_growth": {
            "en": "Marketing tests",
            "fr": "Tests marketing",
            "ar": "اختبارات تسويقية",
        },
        "cost_note_operations": {
            "en": "Hiring or process redesign",
            "fr": "Recrutement ou refonte des processus",
            "ar": "توظيف أو إعادة تصميم العمليات",
        },
        "cost_note_timing": {
            "en": "Opportunity cost of delay",
            "fr": "Coût d'opportunité du retard",
            "ar": "تكلفة الفرصة البديلة للتأخير",
        },
        "cost_note_audience": {
            "en": "Research time",
            "fr": "Temps de recherche",
            "ar": "وقت البحث",
        },
        "gtm_b2c": {
            "en": "Start with a narrow B2C beachhead in {location}; use local communities and referrals.",
            "fr": "Démarrez avec une tête de pont B2C étroite à {location} ; utilisez les communautés locales et le bouche-à-oreille.",
            "ar": "ابدأ بشريحة B2C ضيقة في {location} واستعمل المجتمعات المحلية والتوصيات.",
        },
        "gtm_b2b": {
            "en": "Build 10 pilot B2B accounts in {location} and sell the measurable ROI, not features.",
            "fr": "Construisez 10 comptes B2B pilotes à {location} et vendez le ROI mesurable, pas les fonctionnalités.",
            "ar": "ابنِ 10 حسابات B2B تجريبية في {location} وبِع العائد القابل للقياس لا الميزات.",
        },
        "gtm_b2b2c": {
            "en": "Recruit a few B2B partners in {location} who already own the end customer relationship.",
            "fr": "Recrutez quelques partenaires B2B à {location} qui possèdent déjà la relation client final.",
            "ar": "اجذب شركاء B2B في {location} يملكون علاقة مباشرة مع العميل النهائي.",
        },
        "gtm_marketplace": {
            "en": "Solve the chicken-and-egg problem by manually seeding the scarce side in {location}.",
            "fr": "Résolvez le problème de la poule et de l'œuf en amorçant manuellement le côté rare à {location}.",
            "ar": "حل مشكلة البداية بتغذية الجانب الأصعب يدويًا في {location}.",
        },
        "gtm_validate": {
            "en": "Run a 4-week concierge/MVP test before spending the full budget.",
            "fr": "Lancez un test concierge/MVP de 4 semaines avant de dépenser tout le budget.",
            "ar": "نفّذ اختبار MVP أو خدمة مرافقة لمدة 4 أسابيع قبل صرف كل الميزانية.",
        },
        "gtm_niche": {
            "en": "Win one narrow niche first instead of competing head-on.",
            "fr": "Gagnez d'abord une niche étroite au lieu d'attaquer de front.",
            "ar": "اكسب شريحة ضيقة أولاً بدل المواجهة المباشرة.",
        },
        "gtm_growth": {
            "en": "Use the favorable market trend to test paid acquisition early.",
            "fr": "Profitez de la tendance favorable pour tester l'acquisition payante tôt.",
            "ar": "استغل اتجاه السوق الإيجابي لاختبار الاكتساب المدفوع مبكرًا.",
        },
        "gtm_subscription": {
            "en": "Offer a low-risk trial and track activation, retention and payback.",
            "fr": "Proposez un essai à faible risque et suivez l'activation, la rétention et le payback.",
            "ar": "قدّم تجربة منخفضة المخاطر وتابع التنشيط والاحتفاظ وفترة الاسترداد.",
        },
        "segment_b2c": {
            "en": "Urban B2C customers in {location} who feel the {sector} pain frequently and value convenience.",
            "fr": "Clients B2C urbains à {location} qui vivent souvent le problème {sector} et valorisent la simplicité.",
            "ar": "عملاء B2C حضريون في {location} يعانون من مشكلة {sector} ويقدّرون السهولة.",
        },
        "segment_b2b": {
            "en": "SMEs in {location} with a clear operational pain, a budget, and a fast decision maker.",
            "fr": "PME à {location} avec un problème opérationnel clair, un budget et un décideur rapide.",
            "ar": "شركات صغيرة في {location} لديها مشكلة تشغيلية واضحة وميزانية وصاحب قرار سريع.",
        },
        "segment_b2b2c": {
            "en": "B2B intermediaries in {location} whose end customers need the {sector} offer.",
            "fr": "Intermédiaires B2B à {location} dont les clients finaux ont besoin de l'offre {sector}.",
            "ar": "وسطاء B2B في {location} يحتاج عملاؤهم النهائيون إلى عرض {sector}.",
        },
        "segment_marketplace": {
            "en": "The scarce side of the marketplace in {location} first, then the paying side.",
            "fr": "D'abord le côté rare de la marketplace à {location}, puis le côté payant.",
            "ar": "الجانب الأصعب من السوق في {location} أولاً ثم الجانب الذي يدفع.",
        },
        "your_market": {"en": "your market", "fr": "votre marché", "ar": "سوقك"},
        "bench_net_margin": {"en": "Net margin", "fr": "Marge nette", "ar": "الهامش الصافي"},
        "bench_growth": {"en": "Monthly growth", "fr": "Croissance mensuelle", "ar": "النمو الشهري"},
        "bench_churn": {"en": "Monthly churn", "fr": "Attrition mensuelle", "ar": "التسرب الشهري"},
        "bench_cac": {"en": "Customer acquisition cost", "fr": "Coût d'acquisition client", "ar": "تكلفة اكتساب العميل"},
        "bench_ltv_cac": {"en": "LTV / CAC", "fr": "LTV / CAC", "ar": "LTV / CAC"},
        "bench_marketing_ratio": {"en": "Marketing / revenue", "fr": "Marketing / CA", "ar": "التسويق / الإيرادات"},
        "problem_negative_profit_title": {"en": "The business is not profitable", "fr": "L'entreprise n'est pas rentable", "ar": "المشروع غير مربح"},
        "problem_negative_profit_detail": {
            "en": "The latest monthly profit is {profit:.0f}. Fixed costs, margin or volume must be corrected.",
            "fr": "Le dernier profit mensuel est de {profit:.0f}. Il faut corriger les coûts fixes, la marge ou le volume.",
            "ar": "الربح الشهري الأخير {profit:.0f}. يجب تصحيح التكاليف الثابتة أو الهامش أو الحجم.",
        },
        "problem_low_margin_title": {"en": "Net margin below the sector benchmark", "fr": "Marge nette sous la référence sectorielle", "ar": "الهامش الصافي أقل من المعيار القطاعي"},
        "problem_low_margin_detail": {
            "en": "Net margin is {margin:.1f}% versus a {bench_margin:.1f}% benchmark. Pricing and cost structure need review.",
            "fr": "La marge nette est de {margin:.1f}% contre {bench_margin:.1f}% de référence. Le prix et les coûts doivent être revus.",
            "ar": "الهامش الصافي {margin:.1f}% مقابل معيار {bench_margin:.1f}%. يجب مراجعة التسعير وهيكل التكاليف.",
        },
        "problem_declining_title": {"en": "Revenue trend is declining", "fr": "La tendance du chiffre d'affaires baisse", "ar": "اتجاه الإيرادات في تراجع"},
        "problem_declining_detail": {
            "en": "Estimated monthly growth is {growth:.1f}%. Reactivate customers and diagnose the acquisition funnel.",
            "fr": "La croissance mensuelle estimée est de {growth:.1f}%. Réactivez les clients et analysez le tunnel d'acquisition.",
            "ar": "النمو الشهري المقدر {growth:.1f}%. أعد تنشيط العملاء وحلل مسار الاكتساب.",
        },
        "problem_high_churn_title": {"en": "Customer churn is too high", "fr": "L'attrition client est trop élevée", "ar": "تسرب العملاء مرتفع جدًا"},
        "problem_high_churn_detail": {
            "en": "Churn is {churn:.1f}% versus {bench_churn:.1f}% for the sector. Retention is leaking revenue.",
            "fr": "L'attrition est de {churn:.1f}% contre {bench_churn:.1f}% pour le secteur. La rétention fait fuir le CA.",
            "ar": "التسرب {churn:.1f}% مقابل {bench_churn:.1f}% للقطاع. الاحتفاظ بالعملاء يسرّب الإيرادات.",
        },
        "problem_ltv_cac_title": {"en": "Acquisition economics are weak", "fr": "L'économie d'acquisition est faible", "ar": "اقتصاديات الاكتساب ضعيفة"},
        "problem_ltv_cac_detail": {
            "en": "LTV/CAC is {ltv_cac:.1f}×. Aim for at least 3× to fund growth safely.",
            "fr": "Le LTV/CAC est de {ltv_cac:.1f}×. Visez au moins 3× pour financer la croissance.",
            "ar": "LTV/CAC هو {ltv_cac:.1f}×. استهدف 3× على الأقل لتمويل النمو بأمان.",
        },
        "problem_high_marketing_title": {"en": "Marketing spend is disproportionate", "fr": "Les dépenses marketing sont disproportionnées", "ar": "مصروف التسويق غير متناسب"},
        "problem_high_marketing_detail": {
            "en": "Marketing represents {marketing_ratio:.1f}% of revenue versus {bench_marketing:.1f}% benchmark.",
            "fr": "Le marketing représente {marketing_ratio:.1f}% du CA contre {bench_marketing:.1f}% de référence.",
            "ar": "التسويق يمثل {marketing_ratio:.1f}% من الإيرادات مقابل معيار {bench_marketing:.1f}%.",
        },
        "problem_volatility_title": {"en": "Revenue is volatile", "fr": "Le chiffre d'affaires est volatil", "ar": "الإيرادات متقلبة"},
        "problem_volatility_detail": {
            "en": "Revenue volatility is {volatility:.0f}%. Diversify channels and build recurring revenue.",
            "fr": "La volatilité du CA est de {volatility:.0f}%. Diversifiez les canaux et développez le revenu récurrent.",
            "ar": "تقلب الإيرادات {volatility:.0f}%. نوّع القنوات وابنِ إيرادات متكررة.",
        },
        "problem_cost_growth_title": {"en": "Costs are growing faster than revenue", "fr": "Les coûts augmentent plus vite que le CA", "ar": "التكاليف تنمو أسرع من الإيرادات"},
        "problem_cost_growth_detail": {
            "en": "Cost growth is {cost_growth:.1f}% versus revenue growth of {revenue_growth:.1f}%.",
            "fr": "La croissance des coûts est de {cost_growth:.1f}% contre {revenue_growth:.1f}% du CA.",
            "ar": "نمو التكاليف {cost_growth:.1f}% مقابل نمو الإيرادات {revenue_growth:.1f}%.",
        },
        "business_action_profit": {
            "en": "Rebalance price, variable costs and fixed costs to reach positive unit economics",
            "fr": "Rééquilibrer prix, coûts variables et coûts fixes pour une économie unitaire positive",
            "ar": "إعادة موازنة السعر والتكاليف المتغيرة والثابتة لتحقيق اقتصاديات وحدة موجبة",
        },
        "business_action_margin": {
            "en": "Increase price by 5–8% and renegotiate the two largest cost lines",
            "fr": "Augmenter le prix de 5–8% et renégocier les deux plus gros postes de coûts",
            "ar": "رفع السعر 5–8% وإعادة التفاوض على أكبر بندي تكلفة",
        },
        "business_action_growth": {
            "en": "Run a win-back campaign and fix the lowest-converting acquisition step",
            "fr": "Lancer une campagne de réactivation et corriger l'étape d'acquisition la moins performante",
            "ar": "تشغيل حملة استعادة العملاء وإصلاح أضعف خطوة في الاكتساب",
        },
        "business_action_retention": {
            "en": "Interview churned customers and add onboarding + 30-day retention checkpoints",
            "fr": "Interroger les clients perdus et ajouter onboarding + points de rétention à 30 jours",
            "ar": "مقابلة العملاء المتسربين وإضافة تأهيل ونقاط متابعة للاحتفاظ بعد 30 يومًا",
        },
        "business_action_cac": {
            "en": "Cut the weakest acquisition channel and negotiate better ad rates",
            "fr": "Couper le canal d'acquisition le plus faible et négocier de meilleurs tarifs",
            "ar": "إيقاف أضعف قناة اكتساب والتفاوض على أسعار إعلانات أفضل",
        },
        "business_action_marketing": {
            "en": "Rebalance marketing toward channels with measurable payback",
            "fr": "Réallouer le marketing vers les canaux à payback mesurable",
            "ar": "إعادة توزيع التسويق نحو القنوات ذات الاسترداد القابل للقياس",
        },
        "business_action_cost": {
            "en": "Audit costs and renegotiate recurring suppliers and overhead",
            "fr": "Auditer les coûts et renégocier les fournisseurs récurrents et les frais généraux",
            "ar": "تدقيق التكاليف وإعادة التفاوض مع الموردين والمصاريف المتكررة",
        },
        "business_action_forecast": {
            "en": "Build a 13-week cash-flow forecast and diversify revenue sources",
            "fr": "Construire une prévision de trésorerie à 13 semaines et diversifier les revenus",
            "ar": "بناء توقعات سيولة لـ 13 أسبوعًا وتنويع مصادر الإيراد",
        },
        "business_action_optimize": {
            "en": "Continue disciplined testing: price, retention and acquisition payback",
            "fr": "Continuer les tests disciplinés : prix, rétention et payback d'acquisition",
            "ar": "مواصلة الاختبار المنظم: السعر والاحتفاظ واسترداد تكلفة الاكتساب",
        },
        "business_rationale_negative_profit": {
            "en": "Restoring positive unit economics is the prerequisite for any growth spend.",
            "fr": "Rétablir une économie unitaire positive est le prérequis à toute dépense de croissance.",
            "ar": "استعادة اقتصاديات وحدة موجبة شرط لأي إنفاق على النمو.",
        },
        "business_rationale_low_margin": {
            "en": "A 1–2 point margin gain compounds directly into cash and valuation.",
            "fr": "Un gain de 1–2 points de marge se transforme directement en cash et valeur.",
            "ar": "تحسين الهامش بنقطة أو نقطتين ينعكس مباشرة على النقد والقيمة.",
        },
        "business_rationale_declining": {
            "en": "Stopping the decline is usually cheaper than acquiring entirely new demand.",
            "fr": "Stopper la baisse coûte généralement moins cher que d'acquérir une demande entièrement nouvelle.",
            "ar": "وقف التراجع عادة أرخص من اكتساب طلب جديد بالكامل.",
        },
        "business_rationale_high_churn": {
            "en": "Retention protects revenue and improves LTV without extra acquisition budget.",
            "fr": "La rétention protège le CA et améliore la LTV sans budget d'acquisition supplémentaire.",
            "ar": "الاحتفاظ يحمي الإيرادات ويحسن LTV دون ميزانية اكتساب إضافية.",
        },
        "business_rationale_ltv_cac": {
            "en": "At LTV/CAC below 3×, paid growth destroys cash.",
            "fr": "Avec un LTV/CAC sous 3×, la croissance payante détruit du cash.",
            "ar": "عندما يكون LTV/CAC أقل من 3× فإن النمو المدفوع يستهلك النقد.",
        },
        "business_rationale_high_marketing": {
            "en": "Reallocating spend usually improves CAC without reducing volume.",
            "fr": "Réallouer les dépenses améliore souvent le CAC sans réduire le volume.",
            "ar": "إعادة توزيع الإنفاق تحسن CAC عادة دون تقليل الحجم.",
        },
        "business_rationale_cost_growth": {
            "en": "Cost discipline protects margin as the business scales.",
            "fr": "La discipline sur les coûts protège la marge pendant la croissance.",
            "ar": "ضبط التكاليف يحمي الهامش أثناء النمو.",
        },
        "business_rationale_volatility": {
            "en": "Recurring and diversified revenue lowers forecasting and cash risk.",
            "fr": "Un revenu récurrent et diversifié réduit le risque de prévision et de trésorerie.",
            "ar": "الإيراد المتكرر والمنوع يقلل مخاطر التوقعات والسيولة.",
        },
        "business_rationale_optimize": {
            "en": "Small controlled experiments keep improving the unit economics.",
            "fr": "De petites expériences contrôlées améliorent continuellement l'économie unitaire.",
            "ar": "التجارب الصغيرة المنضبطة تحسن اقتصاديات الوحدة باستمرار.",
        },
        "quality_cost_estimated": {
            "en": "Costs were missing, so they were estimated at 72% of revenue.",
            "fr": "Les coûts manquaient ; ils ont été estimés à 72% du CA.",
            "ar": "كانت التكاليف ناقصة، لذا قُدرت بـ 72% من الإيرادات.",
        },
        "quality_date_generated": {
            "en": "No date column was found; monthly periods were generated automatically.",
            "fr": "Aucune colonne de date trouvée ; les périodes mensuelles ont été générées automatiquement.",
            "ar": "لم يوجد عمود تاريخ؛ تم توليد الفترات الشهرية تلقائيًا.",
        },
        "quality_customers_missing": {
            "en": "Customer counts were missing, so LTV/churn are partly benchmark-based.",
            "fr": "Le nombre de clients manquait ; LTV/attrition reposent en partie sur la référence.",
            "ar": "عدد العملاء ناقص، لذا LTV والتسرب مبنيان جزئيًا على المعيار.",
        },
        "quality_marketing_missing": {
            "en": "Marketing spend was missing; CAC cannot be measured reliably.",
            "fr": "Les dépenses marketing manquaient ; le CAC ne peut pas être mesuré de façon fiable.",
            "ar": "مصروف التسويق ناقص؛ لا يمكن قياس CAC بدقة.",
        },
        "quality_manual_snapshot": {
            "en": "This analysis uses a manual snapshot and an assumed constant monthly growth.",
            "fr": "Cette analyse utilise un instantané manuel et une croissance mensuelle supposée constante.",
            "ar": "هذا التحليل يستعمل لقطة يدوية ونموًا شهريًا ثابتًا مفترضًا.",
        },
        "business_summary": {
            "en": "Last month: revenue {revenue} {currency}, profit {profit} {currency}, monthly growth {growth}%.",
            "fr": "Dernier mois : CA {revenue} {currency}, profit {profit} {currency}, croissance mensuelle {growth}%.",
            "ar": "الشهر الأخير: إيرادات {revenue} {currency}، ربح {profit} {currency}، نمو شهري {growth}%.",
        },
        "monthly_snapshot_hint": {
            "en": "A snapshot is converted into 12 indicative months ending today, then forecast.",
            "fr": "Un instantané est converti en 12 mois indicatifs jusqu'à aujourd'hui, puis projeté.",
            "ar": "تُحوَّل اللقطة إلى 12 شهرًا تقديريًا حتى اليوم ثم تُتوقع.",
        },
        "business_insight": {
            "en": "The KPI and benchmark view is diagnostic; the forecast is statistical, not a promise.",
            "fr": "La vue KPI et références est diagnostique ; la prévision est statistique, pas une promesse.",
            "ar": "عرض المؤشرات والمعايير تشخيصي؛ التوقع إحصائي وليس وعدًا.",
        },
        "report_title_idea": {
            "en": "Business idea evaluation report",
            "fr": "Rapport d'évaluation d'idée",
            "ar": "تقرير تقييم فكرة مشروع",
        },
        "report_title_business": {
            "en": "Business performance report",
            "fr": "Rapport de performance",
            "ar": "تقرير أداء المشروع",
        },
        "generated_on": {"en": "Generated on", "fr": "Généré le", "ar": "تم التوليد في"},
        "criteria": {"en": "Criterion", "fr": "Critère", "ar": "المعيار"},
        "weight": {"en": "Weight", "fr": "Poids", "ar": "الوزن"},
        "scenario": {"en": "Scenario", "fr": "Scénario", "ar": "السيناريو"},
        "financial_assumptions": {
            "en": "Financial assumptions",
            "fr": "Hypothèses financières",
            "ar": "الافتراضات المالية",
        },
        "key_metrics": {"en": "Key metrics", "fr": "Indicateurs clés", "ar": "المؤشرات الرئيسية"},
        "chart_scores": {"en": "Criterion scores (out of 10)", "fr": "Scores par critère (sur 10)", "ar": "درجات المعايير (من 10)"},
        "chart_forecast": {"en": "Revenue forecast", "fr": "Prévision du chiffre d'affaires", "ar": "توقع الإيرادات"},
        "chart_monthly_profit": {"en": "Monthly profit", "fr": "Profit mensuel", "ar": "الربح الشهري"},
        "pdf_note": {
            "en": "ML estimates and scenarios are decision support, not financial advice.",
            "fr": "Les estimations ML et scénarios sont une aide à la décision, pas un conseil financier.",
            "ar": "تقديرات التعلم الآلي والسيناريوهات دعم للقرار وليست نصيحة مالية.",
        },
        "analysis_method": {
            "en": "Method: scikit-learn classifier + regression + transparent rubric.",
            "fr": "Méthode : classifieur scikit-learn + régression + grille transparente.",
            "ar": "الطريقة: مصنف scikit-learn + انحدار + معايير شفافة.",
        },
    }
)


def t(key: str, language: str = "en", **kwargs: object) -> str:
    """Translate a key, falling back to English, then to the key itself."""
    language = normalize_language(language)
    entry = TRANSLATIONS.get(key)
    if not entry:
        return key
    text = entry.get(language) or entry.get("en") or key
    if kwargs:
        try:
            return text.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            return text
    return text
