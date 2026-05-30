"""
Semantic Chunker Service — Final Retrieval-Aware Architecture
Converts a ParsedDocument into a list of TextChunk objects.
Preserves procedures, detects troubleshooting blocks, and assigns rich routing metadata.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

try:
    import tiktoken
    _tokenizer = tiktoken.get_encoding("cl100k_base")
    _USE_TIKTOKEN = True
except ImportError:
    _USE_TIKTOKEN = False

from app.configs.config import settings
from app.models.schemas import TextChunk, QueryIntent
from app.services.pdf_parser import ParsedDocument, ParsedPage, TextBlock
from app.utils.logger import setup_logger

logger = setup_logger("maritimemind.chunker")

_PROCEDURE_PATTERN = re.compile(
    r"^\s*("                        # Start of line
    r"\d+[\.)\-]\s+"                # 1. or 1) or 1-
    r"|[a-zA-Z][\.)\-]\s+"         # a. or b) or A.
    r"|[ivxlIVXL]+[\.)\-]\s+"      # i. ii. iii. iv. (roman numerals)
    r"|\-\s+(?=[A-Z])"             # - followed by capital letter
    r")",
    re.MULTILINE
)
_TABLE_MARKER = re.compile(r"^\|.+\|", re.MULTILINE)
_WARNING_MARKER = re.compile(r"(?i)\b(warning|caution|danger|note|important)\b")
_EMERGENCY_MARKER = re.compile(r"(?i)\b(emergency|fire|abandon|collision|grounding|mayday|distress)\b")
_TROUBLESHOOTING_MARKER = re.compile(r"(?i)\b(symptom|cause|fault|troubleshoot|remedy|defect|corrective|probable)\b")
_MAINTENANCE_MARKER = re.compile(r"(?i)\b(overhaul|dismantl|assembl|disassembl|inspection|maintenance|recondit)\b")
_DIAGRAM_MARKER = re.compile(r"(?i)\b(figure|fig\.?|diagram|schematic|see below|drawing)\b")

class SemanticChunkerService:
    def __init__(self) -> None:
        self._chunk_size = settings.CHUNK_SIZE
        self._overlap = settings.CHUNK_OVERLAP
        self._min_length = settings.MIN_CHUNK_LENGTH

    def chunk_document(
        self,
        parsed_doc: ParsedDocument,
        department: str = "general",
        ship_id: Optional[str] = None,
        language: Optional[str] = None,
    ) -> List[TextChunk]:
        chunks: List[TextChunk] = []
        hierarchy_stack: List[str] = []
        chunk_counter = 1
        
        for page in parsed_doc.pages:
            # Tables
            for table_md in page.tables:
                if not table_md.strip(): continue
                section_title = hierarchy_stack[-1] if hierarchy_stack else "Table"
                subsystem = self._derive_subsystem(hierarchy_stack)
                content = f"[TABLE — {section_title}]\n\n{table_md}"
                
                chunk = self._make_chunk(
                    content=content,
                    manual_name=parsed_doc.manual_name,
                    department=department,
                    subsystem=subsystem,
                    page_number=page.page_number,
                    hierarchy_path=list(hierarchy_stack),
                    section_title=section_title,
                    ship_id=ship_id,
                    language=language,
                    chunk_index=chunk_counter
                )
                if chunk:
                    chunks.append(chunk)
                    chunk_counter += 1

            # Text blocks
            accumulated_text = ""
            for tb in page.text_blocks:
                if not tb.text.strip(): continue

                if self._is_heading(tb):
                    if accumulated_text.strip():
                        new_chunks = self._flush_text(
                            accumulated_text, parsed_doc.manual_name, department,
                            page.page_number, hierarchy_stack, ship_id, language, chunk_counter
                        )
                        chunks.extend(new_chunks)
                        chunk_counter += len(new_chunks)
                        accumulated_text = ""
                    
                    heading_level = self._estimate_heading_level(tb)
                    hierarchy_stack = hierarchy_stack[:heading_level - 1]
                    hierarchy_stack.append(tb.text.strip())
                    continue

                if self._is_procedure(tb.text) or self._is_troubleshooting(tb.text):
                    if accumulated_text.strip():
                        new_chunks = self._flush_text(
                            accumulated_text, parsed_doc.manual_name, department,
                            page.page_number, hierarchy_stack, ship_id, language, chunk_counter
                        )
                        chunks.extend(new_chunks)
                        chunk_counter += len(new_chunks)
                        accumulated_text = ""
                    
                    # Ensure procedural blocks are kept mostly intact
                    proc_chunks = self._chunk_procedure(
                        tb.text, parsed_doc.manual_name, department,
                        page.page_number, hierarchy_stack, ship_id, language, chunk_counter
                    )
                    chunks.extend(proc_chunks)
                    chunk_counter += len(proc_chunks)
                    continue

                accumulated_text += " " + tb.text
                
                # Use a larger semantic window before splitting body text
                if self._token_count(accumulated_text) >= self._chunk_size + 200:
                    new_chunks = self._flush_text(
                        accumulated_text, parsed_doc.manual_name, department,
                        page.page_number, hierarchy_stack, ship_id, language, chunk_counter
                    )
                    chunks.extend(new_chunks)
                    chunk_counter += len(new_chunks)
                    accumulated_text = self._last_n_tokens(accumulated_text, self._overlap)

            if accumulated_text.strip():
                new_chunks = self._flush_text(
                    accumulated_text, parsed_doc.manual_name, department,
                    page.page_number, hierarchy_stack, ship_id, language, chunk_counter
                )
                chunks.extend(new_chunks)
                chunk_counter += len(new_chunks)
                
        # Build linked chains (Contextual Expansion)
        chunks = self._link_chunks(chunks)
        
        logger.info(f"Chunked '{parsed_doc.manual_name}': {len(chunks)} semantic chunks generated.")
        return chunks

    def _flush_text(
        self, text: str, manual_name: str, department: str, page: int, 
        hierarchy: List[str], ship_id: Optional[str], language: Optional[str], start_idx: int
    ) -> List[TextChunk]:
        segments = self._split_with_overlap(text)
        chunks = []
        for i, segment in enumerate(segments):
            chunk = self._make_chunk(
                content=segment,
                manual_name=manual_name,
                department=department,
                subsystem=self._derive_subsystem(hierarchy),
                page_number=page,
                hierarchy_path=list(hierarchy),
                section_title=hierarchy[-1] if hierarchy else "General",
                ship_id=ship_id,
                language=language,
                chunk_index=start_idx + i
            )
            if chunk: chunks.append(chunk)
        return chunks

    def _chunk_procedure(
        self, text: str, manual_name: str, department: str, page: int, 
        hierarchy: List[str], ship_id: Optional[str], language: Optional[str], start_idx: int
    ) -> List[TextChunk]:
        max_size = self._chunk_size * 2
        if self._token_count(text) <= max_size:
            chunk = self._make_chunk(
                content=text, manual_name=manual_name, department=department,
                subsystem=self._derive_subsystem(hierarchy), page_number=page,
                hierarchy_path=list(hierarchy), section_title=hierarchy[-1] if hierarchy else "Procedure",
                ship_id=ship_id, language=language, chunk_index=start_idx
            )
            return [chunk] if chunk else []
            
        step_pattern = re.compile(r"(?=^\s*\d+[\.\)]\s+)", re.MULTILINE)
        steps = step_pattern.split(text)
        chunks, current = [], ""
        idx = start_idx
        
        for step in steps:
            if not step.strip(): continue
            if self._token_count(current + step) > max_size and current.strip():
                chunk = self._make_chunk(
                    content=current, manual_name=manual_name, department=department,
                    subsystem=self._derive_subsystem(hierarchy), page_number=page,
                    hierarchy_path=list(hierarchy), section_title=hierarchy[-1] if hierarchy else "Procedure",
                    ship_id=ship_id, language=language, chunk_index=idx
                )
                if chunk: 
                    chunks.append(chunk)
                    idx += 1
                current = step
            else:
                current += step

        if current.strip():
            chunk = self._make_chunk(
                content=current, manual_name=manual_name, department=department,
                subsystem=self._derive_subsystem(hierarchy), page_number=page,
                hierarchy_path=list(hierarchy), section_title=hierarchy[-1] if hierarchy else "Procedure",
                ship_id=ship_id, language=language, chunk_index=idx
            )
            if chunk: chunks.append(chunk)
        return chunks

    def _split_with_overlap(self, text: str) -> List[str]:
        text = text.strip()
        if self._token_count(text) <= self._chunk_size:
            return [text] if len(text) >= self._min_length else []
        sentence_end = re.compile(r'(?<=[.!?])\s+')
        sentences = sentence_end.split(text)
        segments, current_sentences = [], []
        current_len = 0
        for sentence in sentences:
            s_len = self._token_count(sentence)
            if current_len + s_len > self._chunk_size and current_sentences:
                segments.append(" ".join(current_sentences))
                # Overlap: keep last N tokens worth of sentences
                overlap_sentences = []
                overlap_len = 0
                for sent in reversed(current_sentences):
                    sent_tok = self._token_count(sent)
                    if overlap_len + sent_tok <= self._overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_len += sent_tok
                    else:
                        break
                current_sentences = overlap_sentences + [sentence]
                current_len = self._token_count(" ".join(current_sentences))
            else:
                current_sentences.append(sentence)
                current_len += s_len
        if current_sentences:
            remainder = " ".join(current_sentences)
            if len(remainder) >= self._min_length:
                segments.append(remainder)
        return segments

    def _make_chunk(
        self, content: str, manual_name: str, department: str, subsystem: str,
        page_number: int, hierarchy_path: List[str], section_title: str,
        ship_id: Optional[str], language: Optional[str], chunk_index: int
    ) -> Optional[TextChunk]:
        content = content.strip()
        if len(content) < self._min_length: return None

        # Readable semantic ID format
        clean_dept = department[:3].lower()
        clean_sub = subsystem.replace(" ", "_").lower()
        # strip punctuation from subsystem
        clean_sub = re.sub(r'[^\w\s]', '', clean_sub)[:10]
        chunk_id = f"{clean_dept}_{clean_sub}_p{page_number:03d}_c{chunk_index:03d}"

        is_proc = self._is_procedure(content)
        is_maint = bool(_MAINTENANCE_MARKER.search(content))
        is_warn = bool(_WARNING_MARKER.search(content))
        is_emerg = bool(_EMERGENCY_MARKER.search(content))
        is_diag_ref = bool(_DIAGRAM_MARKER.search(content))
        is_trouble = self._is_troubleshooting(content)
        
        intents = []
        if is_proc or is_maint: intents.append(QueryIntent.PROCEDURE)
        if is_trouble: intents.append(QueryIntent.TROUBLESHOOTING)
        if is_emerg: intents.append(QueryIntent.EMERGENCY)
        if is_diag_ref: intents.append(QueryIntent.DIAGRAM_REQUEST)
        if not intents: intents.append(QueryIntent.EXPLANATION)

        importance = "high" if (is_warn or is_emerg or is_trouble) else ("medium" if (is_proc or is_maint) else "low")

        return TextChunk(
            chunk_id=chunk_id,
            manual_name=manual_name,
            ship_id=ship_id,
            language=language,
            department=department,
            subsystem=subsystem,
            document_type="manual",
            page_number=page_number,
            section_title=section_title,
            content=content,
            contains_procedure=is_proc,
            contains_warning=is_warn,
            contains_emergency_workflow=is_emerg,
            contains_diagram_reference=is_diag_ref,
            importance=importance,
            applicable_intents=intents,
            hierarchy_path=hierarchy_path,
            embedding_model=settings.TEXT_EMBEDDING_MODEL,
        )

    @staticmethod
    def _link_chunks(chunks: List[TextChunk]) -> List[TextChunk]:
        for i, chunk in enumerate(chunks):
            chunk.previous_chunk_id = chunks[i - 1].chunk_id if i > 0 else None
            chunk.next_chunk_id = chunks[i + 1].chunk_id if i < len(chunks) - 1 else None
        return chunks

    @staticmethod
    def _derive_subsystem(hierarchy: List[str]) -> str:
        if len(hierarchy) > 1: return hierarchy[1]
        if len(hierarchy) == 1: return hierarchy[0]
        return "general"

    @staticmethod
    def _is_heading(tb: TextBlock) -> bool:
        return (tb.font_size >= 11.0 or bool(tb.font_flags & 16)) and len(tb.text.strip()) < 120

    @staticmethod
    def _estimate_heading_level(tb: TextBlock) -> int:
        if tb.font_size >= 18: return 1
        elif tb.font_size >= 14: return 2
        elif tb.font_size >= 12: return 3
        return 4

    @staticmethod
    def _is_procedure(text: str) -> bool:
        return len(_PROCEDURE_PATTERN.findall(text)) >= 2

    @staticmethod
    def _is_troubleshooting(text: str) -> bool:
        return bool(_TROUBLESHOOTING_MARKER.search(text))

    @staticmethod
    def _token_count(text: str) -> int:
        """Accurate token count using tiktoken (cl100k_base encoding).
        Falls back to whitespace split if tiktoken is not installed.
        """
        if _USE_TIKTOKEN:
            return len(_tokenizer.encode(text))
        return len(text.split())

    @staticmethod
    def _last_n_tokens(text: str, n: int) -> str:
        """Return the last n tokens of text. Uses tiktoken when available."""
        if _USE_TIKTOKEN:
            tokens = _tokenizer.encode(text)
            return _tokenizer.decode(tokens[-n:]) if len(tokens) > n else text
        words = text.split()
        return " ".join(words[-n:]) if len(words) > n else text
