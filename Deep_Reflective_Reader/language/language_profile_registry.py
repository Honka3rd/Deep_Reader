import re
from dataclasses import dataclass

from language.language_code import LanguageCode, LanguageCodeResolver


@dataclass(frozen=True)
class LanguageProfile:
    """Centralized language profile for runtime language-dependent rules."""
    code: LanguageCode
    global_scope_keywords: tuple[str, ...]
    local_reference_signals: tuple[str, ...]
    detector_labels: tuple[str, ...]


class LanguageProfileRegistry:
    """Global registry for language-specific runtime configuration."""

    _PROFILES: dict[LanguageCode, LanguageProfile] = {
        LanguageCode.EN: LanguageProfile(
            code=LanguageCode.EN,
            global_scope_keywords=(
                "which",
                "what are the main",
                "major",
                "all",
                "list",
                "overall",
                "themes",
                "relationships",
                "reasons",
                "summarize",
                "in the whole book",
                "across the document",
            ),
            local_reference_signals=(
                "this paragraph",
                "that paragraph",
                "this sentence",
                "that sentence",
                "the passage",
                "this section",
                "here",
                "above",
                "below",
            ),
            detector_labels=("english",),
        ),
        LanguageCode.ZH: LanguageProfile(
            code=LanguageCode.ZH,
            global_scope_keywords=(
                "哪些",
                "主要",
                "總結",
                "总结",
                "整體",
                "整体",
                "全書",
                "全书",
                "整篇",
                "原因有哪些",
                "關係有哪些",
                "关系有哪些",
                "主題",
                "主题",
                "列舉",
                "列举",
                "歸納",
                "归纳",
            ),
            local_reference_signals=(
                "這段",
                "这段",
                "這句",
                "这句",
                "這一段",
                "这一段",
                "上一段",
                "下一段",
                "上一句",
                "下一句",
                "這裡",
                "这里",
                "這部分",
                "这部分",
            ),
            detector_labels=("chinese",),
        ),
        LanguageCode.JA: LanguageProfile(
            code=LanguageCode.JA,
            global_scope_keywords=("どの", "主な", "全体", "全書", "一覧", "要約", "テーマ", "関係", "理由"),
            local_reference_signals=("この段落", "この文", "前の段落", "次の段落", "ここ"),
            detector_labels=("japanese",),
        ),
        LanguageCode.KO: LanguageProfile(
            code=LanguageCode.KO,
            global_scope_keywords=("어떤", "주요", "전체", "전반", "목록", "요약", "주제", "관계", "이유"),
            local_reference_signals=("이 단락", "이 문장", "이 부분", "위 문단", "다음 문단", "여기"),
            detector_labels=("korean",),
        ),
        LanguageCode.ES: LanguageProfile(
            code=LanguageCode.ES,
            global_scope_keywords=("cuáles", "principales", "todo", "lista", "general", "temas", "relaciones", "razones", "resumen"),
            local_reference_signals=("este párrafo", "esta oración", "esta frase", "aquí"),
            detector_labels=("spanish",),
        ),
        LanguageCode.PT: LanguageProfile(
            code=LanguageCode.PT,
            global_scope_keywords=("quais", "principais", "tudo", "lista", "geral", "temas", "relacionamentos", "razões", "resumo"),
            local_reference_signals=("este parágrafo", "esta frase", "aqui"),
            detector_labels=("portuguese",),
        ),
        LanguageCode.FR: LanguageProfile(
            code=LanguageCode.FR,
            global_scope_keywords=("quels", "principaux", "tous", "liste", "global", "thèmes", "relations", "raisons", "résumé"),
            local_reference_signals=("ce paragraphe", "cette phrase", "ici"),
            detector_labels=("french",),
        ),
        LanguageCode.DE: LanguageProfile(
            code=LanguageCode.DE,
            global_scope_keywords=("welche", "haupt", "alle", "liste", "insgesamt", "themen", "beziehungen", "gründe", "zusammenfassung"),
            local_reference_signals=("dieser absatz", "dieser satz", "hier"),
            detector_labels=("german",),
        ),
        LanguageCode.IT: LanguageProfile(
            code=LanguageCode.IT,
            global_scope_keywords=("quali", "principali", "tutti", "elenco", "complessivo", "temi", "relazioni", "ragioni", "riassunto"),
            local_reference_signals=("questo paragrafo", "questa frase", "qui"),
            detector_labels=("italian",),
        ),
        LanguageCode.RU: LanguageProfile(
            code=LanguageCode.RU,
            global_scope_keywords=("какие", "основные", "все", "список", "в целом", "темы", "отношения", "причины", "сводка"),
            local_reference_signals=("этот абзац", "это предложение", "здесь"),
            detector_labels=("russian",),
        ),
        LanguageCode.AR: LanguageProfile(
            code=LanguageCode.AR,
            global_scope_keywords=("ما هي الرئيسية", "الرئيسية", "كل", "قائمة", "بشكل عام", "الموضوعات", "العلاقات", "الأسباب", "ملخص"),
            local_reference_signals=("هذه الفقرة", "هذه الجملة", "هنا"),
            detector_labels=("arabic",),
        ),
        LanguageCode.HI: LanguageProfile(
            code=LanguageCode.HI,
            global_scope_keywords=("मुख्य", "कौन से", "सभी", "सूची", "समग्र", "थीम", "संबंध", "कारण", "सारांश"),
            local_reference_signals=("यह पैराग्राफ", "यह वाक्य", "यहाँ"),
            detector_labels=("hindi",),
        ),
        LanguageCode.TR: LanguageProfile(
            code=LanguageCode.TR,
            global_scope_keywords=("hangi", "ana", "tümü", "liste", "genel", "temalar", "ilişkiler", "nedenler", "özet"),
            local_reference_signals=("bu paragraf", "bu cümle", "burada"),
            detector_labels=("turkish",),
        ),
        LanguageCode.NL: LanguageProfile(
            code=LanguageCode.NL,
            global_scope_keywords=("welke", "belangrijkste", "alle", "lijst", "algemeen", "thema's", "relaties", "redenen", "samenvatting"),
            local_reference_signals=("deze paragraaf", "deze zin", "hier"),
            detector_labels=("dutch",),
        ),
        LanguageCode.PL: LanguageProfile(
            code=LanguageCode.PL,
            global_scope_keywords=("które", "główne", "wszystkie", "lista", "ogólnie", "tematy", "relacje", "powody", "podsumowanie"),
            local_reference_signals=("ten akapit", "to zdanie", "tutaj"),
            detector_labels=("polish",),
        ),
        LanguageCode.UK: LanguageProfile(
            code=LanguageCode.UK,
            global_scope_keywords=("які", "основні", "усі", "список", "загалом", "теми", "стосунки", "причини", "підсумок"),
            local_reference_signals=("цей абзац", "це речення", "тут"),
            detector_labels=("ukrainian",),
        ),
        LanguageCode.ID: LanguageProfile(
            code=LanguageCode.ID,
            global_scope_keywords=("yang", "utama", "semua", "daftar", "secara keseluruhan", "tema", "hubungan", "alasan", "ringkasan"),
            local_reference_signals=("paragraf ini", "kalimat ini", "di sini"),
            detector_labels=("indonesian",),
        ),
        LanguageCode.VI: LanguageProfile(
            code=LanguageCode.VI,
            global_scope_keywords=("những", "chính", "tất cả", "liệt kê", "tổng thể", "chủ đề", "mối quan hệ", "lý do", "tóm tắt"),
            local_reference_signals=("đoạn này", "câu này", "ở đây"),
            detector_labels=("vietnamese",),
        ),
        LanguageCode.TH: LanguageProfile(
            code=LanguageCode.TH,
            global_scope_keywords=("อะไรบ้าง", "หลัก", "ทั้งหมด", "รายการ", "โดยรวม", "ธีม", "ความสัมพันธ์", "เหตุผล", "สรุป"),
            local_reference_signals=("ย่อหน้านี้", "ประโยคนี้", "ที่นี่"),
            detector_labels=("thai",),
        ),
    }
    _SESSION_LOCAL_ANCHOR_SIGNALS: dict[LanguageCode, tuple[str, ...]] = {
        LanguageCode.EN: ("here", "in this part"),
        LanguageCode.ZH: ("這裡", "这里", "此處", "此处"),
        LanguageCode.JA: ("ここ",),
        LanguageCode.KO: ("여기",),
        LanguageCode.ES: ("aquí", "aca", "acá"),
        LanguageCode.PT: ("aqui", "cá"),
        LanguageCode.FR: ("ici",),
        LanguageCode.DE: ("hier",),
        LanguageCode.IT: ("qui",),
        LanguageCode.RU: ("здесь",),
        LanguageCode.AR: ("هنا",),
        LanguageCode.HI: ("यहाँ", "यहीं"),
        LanguageCode.TR: ("burada",),
        LanguageCode.NL: ("hier",),
        LanguageCode.PL: ("tutaj",),
        LanguageCode.UK: ("тут",),
        LanguageCode.ID: ("sini", "di sini"),
        LanguageCode.VI: ("đây", "ở đây"),
        LanguageCode.TH: ("ที่นี่",),
    }
    _LOW_VALUE_NOT_FOUND_RESPONSES: dict[LanguageCode, str] = {
        LanguageCode.EN: "Not found",
        LanguageCode.ZH: "未找到相關內容",
        LanguageCode.JA: "関連する内容が見つかりませんでした",
        LanguageCode.KO: "관련 내용을 찾지 못했습니다",
        LanguageCode.ES: "No se encontró contenido relevante",
        LanguageCode.PT: "Nenhum conteúdo relevante foi encontrado",
        LanguageCode.FR: "Aucun contenu pertinent trouvé",
        LanguageCode.DE: "Kein relevanter Inhalt gefunden",
        LanguageCode.IT: "Nessun contenuto pertinente trovato",
        LanguageCode.RU: "Релевантный контент не найден",
        LanguageCode.AR: "لم يتم العثور على محتوى ذي صلة",
        LanguageCode.HI: "संबंधित सामग्री नहीं मिली",
        LanguageCode.TR: "İlgili içerik bulunamadı",
        LanguageCode.NL: "Geen relevante inhoud gevonden",
        LanguageCode.PL: "Nie znaleziono odpowiednich treści",
        LanguageCode.UK: "Релевантний вміст не знайдено",
        LanguageCode.ID: "Konten yang relevan tidak ditemukan",
        LanguageCode.VI: "Không tìm thấy nội dung liên quan",
        LanguageCode.TH: "ไม่พบเนื้อหาที่เกี่ยวข้อง",
    }

    @classmethod
    def get_profile(cls, language: LanguageCode) -> LanguageProfile:
        """Get language profile or fall back to English profile."""
        return cls._PROFILES.get(language, cls._PROFILES[LanguageCode.EN])

    @classmethod
    def get_low_value_not_found_response(cls, language: LanguageCode) -> str:
        """Get localized fallback response for low-value / reject questions."""
        if language == LanguageCode.UNKNOWN:
            return cls._LOW_VALUE_NOT_FOUND_RESPONSES[LanguageCode.EN]
        return cls._LOW_VALUE_NOT_FOUND_RESPONSES.get(
            language,
            cls._LOW_VALUE_NOT_FOUND_RESPONSES[LanguageCode.EN],
        )

    @classmethod
    def get_scope_keywords(cls, language: LanguageCode) -> tuple[str, ...]:
        """Get global-scope keywords for a specific language."""
        return cls.get_profile(language).global_scope_keywords

    @classmethod
    def get_all_scope_keywords(cls) -> tuple[str, ...]:
        """Get deduplicated global-scope keywords across profiles."""
        seen: set[str] = set()
        ordered: list[str] = []
        for profile in cls._PROFILES.values():
            for keyword in profile.global_scope_keywords:
                if keyword in seen:
                    continue
                seen.add(keyword)
                ordered.append(keyword)
        return tuple(ordered)

    @classmethod
    def get_local_reference_signals(cls, language: LanguageCode) -> tuple[str, ...]:
        """Backward-compatible alias for strong local-reference signals."""
        return cls.get_strong_local_reference_signals(language)

    @classmethod
    def get_all_local_reference_signals(cls) -> tuple[str, ...]:
        """Backward-compatible alias for all strong local-reference signals."""
        return cls.get_all_strong_local_reference_signals()

    @classmethod
    def get_strong_local_reference_signals(
        cls,
        language: LanguageCode,
    ) -> tuple[str, ...]:
        """Get high-precision local signals (excluding weak session anchors)."""
        if language == LanguageCode.UNKNOWN:
            return cls.get_all_strong_local_reference_signals()

        profile_signals = cls.get_profile(language).local_reference_signals
        weak_anchor_set = {
            marker.lower()
            for marker in cls.get_weak_session_local_anchor_signals(language)
        }
        if not weak_anchor_set:
            return profile_signals
        return tuple(
            signal
            for signal in profile_signals
            if signal.lower() not in weak_anchor_set
        )

    @classmethod
    def get_all_strong_local_reference_signals(cls) -> tuple[str, ...]:
        """Get deduplicated strong local-reference signals across profiles."""
        seen: set[str] = set()
        ordered: list[str] = []
        for profile in cls._PROFILES.values():
            for signal in cls.get_strong_local_reference_signals(profile.code):
                if signal in seen:
                    continue
                seen.add(signal)
                ordered.append(signal)
        return tuple(ordered)

    @classmethod
    def get_session_local_anchor_signals(cls, language: LanguageCode) -> tuple[str, ...]:
        """Backward-compatible alias for weak session-local anchor signals."""
        return cls.get_weak_session_local_anchor_signals(language)

    @classmethod
    def get_weak_session_local_anchor_signals(
        cls,
        language: LanguageCode,
    ) -> tuple[str, ...]:
        """Get weak local anchor signals that rely on active-session context."""
        if language == LanguageCode.UNKNOWN:
            return cls.get_all_weak_session_local_anchor_signals()
        return cls._SESSION_LOCAL_ANCHOR_SIGNALS.get(language, tuple())

    @classmethod
    def get_all_session_local_anchor_signals(cls) -> tuple[str, ...]:
        """Backward-compatible alias for all weak session-local anchor signals."""
        return cls.get_all_weak_session_local_anchor_signals()

    @classmethod
    def get_all_weak_session_local_anchor_signals(cls) -> tuple[str, ...]:
        """Get deduplicated weak local anchor signals across languages."""
        seen: set[str] = set()
        ordered: list[str] = []
        for signals in cls._SESSION_LOCAL_ANCHOR_SIGNALS.values():
            for signal in signals:
                if signal in seen:
                    continue
                seen.add(signal)
                ordered.append(signal)
        return tuple(ordered)

    @classmethod
    def get_supported_language_codes(cls) -> tuple[str, ...]:
        """Get supported language code values used in detection prompts."""
        return tuple(profile.code.value for profile in cls._PROFILES.values())

    @classmethod
    def normalize_detector_output(cls, raw: str | None) -> LanguageCode:
        """Normalize detector output text to canonical language code enum."""
        if raw is None:
            return LanguageCode.UNKNOWN

        compact = raw.strip().lower()
        if not compact:
            return LanguageCode.UNKNOWN

        compact = compact.replace(".", "")
        compact = compact.split()[0]
        resolved = LanguageCodeResolver.resolve(compact)
        if resolved != LanguageCode.UNKNOWN:
            return resolved

        normalized = re.sub(r"[^a-z-]", "", compact)
        resolved = LanguageCodeResolver.resolve(normalized)
        if resolved != LanguageCode.UNKNOWN:
            return resolved

        for profile in cls._PROFILES.values():
            if normalized in profile.detector_labels:
                return profile.code
        return LanguageCode.UNKNOWN
