import asyncio
from pathlib import Path
from typing import Dict, List, Sequence

import httpx
from llama_index.core import Settings
from llama_index.core.async_utils import run_jobs
from llama_index.core.extractors import (
    KeywordExtractor,
    SummaryExtractor,
    TitleExtractor,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.extractors.entity import EntityExtractor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import BaseNode, Document, TextNode

from src.config import OLLAMA_MAX_CONCURRENT_DOCS

MAX_CHARS_FOR_METADATA = 12_000
MAX_LLM_RETRIES = 3
RETRY_DELAY_SECONDS = 5

FINAL_SUMMARY_PROMPT = """\
Here are several summaries of different parts of a document:
{context_str}

Provide a VERY concise global summary of the combined content in 2-5 sentences.
Focus on:
- global topic
- key entities
- overall purpose

Global Summary:
"""


class CustomSummaryExtractor(SummaryExtractor):
    final_summary_prompt: str = FINAL_SUMMARY_PROMPT
    chunk_size: int = 5

    async def _summarize_batch(self, texts: List[str]) -> str:
        context_str = "\n\n".join(texts)

        summary = await self.llm.apredict(
            PromptTemplate(template=self.prompt_template),
            context_str=context_str,
        )
        return summary.strip()

    async def _tree_summarize(self, nodes: Sequence[BaseNode]) -> str:
        if not nodes:
            return ""

        current_level = [
            node.get_content(metadata_mode=self.metadata_mode)
            for node in nodes
        ]

        while len(current_level) > 1:
            jobs = []

            for i in range(0, len(current_level), self.chunk_size):
                batch = current_level[i : i + self.chunk_size]
                jobs.append(self._summarize_batch(batch))

            current_level = await run_jobs(
                jobs,
                show_progress=self.show_progress,
                workers=self.num_workers,
            )

        final_summary = await self.llm.apredict(
            PromptTemplate(template=self.final_summary_prompt),
            context_str=current_level[0],
        )

        return final_summary.strip()

    async def aextract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        metadata_list = await super().aextract(nodes)

        global_summary = await self._tree_summarize(nodes)

        for metadata in metadata_list:
            metadata["global_summary"] = global_summary

        return metadata_list


def get_ingestion_pipeline() -> IngestionPipeline:
    transformations = [
        #TitleExtractor(nodes=5),
        #QuestionsAnsweredExtractor(questions=3),
        #CustomSummaryExtractor(summaries=["prev", "self"]),
        #KeywordExtractor(keywords=5),
        EntityExtractor(prediction_threshold=0.5),
    ]

    return IngestionPipeline(transformations=transformations)


async def _extract_document_metadata_async(doc: Document) -> dict[str, str]:
    llm = Settings.llm
    if llm is None:
        raise ValueError("Settings.llm must be set before indexing.")

    node = TextNode(
        text=doc.get_content(),
        metadata=doc.metadata,
    )
    title_extractor = TitleExtractor(llm=llm, nodes=1)
    summary_extractor = SummaryExtractor(llm=llm, summaries=["self"])

    for attempt in range(1, MAX_LLM_RETRIES + 1):
        try:
            title_metadata = (await title_extractor.aextract([node]))[0]
            summary_metadata = (await summary_extractor.aextract([node]))[0]
            break
        except httpx.ReadTimeout:
            if attempt == MAX_LLM_RETRIES:
                raise
            await asyncio.sleep(RETRY_DELAY_SECONDS * attempt)

    title = title_metadata.get("document_title", "").strip()
    summary = summary_metadata.get("section_summary", "").strip()

    if not title:
        file_name = doc.metadata.get("file_name") or doc.metadata.get("file_path", "")
        title = Path(file_name).stem if file_name else "Untitled"

    return {
        "document_title": title,
        "document_summary": summary,
    }


def extract_document_metadata(doc: Document) -> dict[str, str]:
    return asyncio.run(_extract_document_metadata_async(doc))


async def _process_document_async(
    doc: Document,
    splitter: SentenceSplitter,
    semaphore: asyncio.Semaphore,
) -> list[BaseNode]:
    async with semaphore:
        doc_metadata = await _extract_document_metadata_async(doc)
        doc_nodes = splitter.get_nodes_from_documents([doc])

        for node in doc_nodes:
            node.metadata.update(doc_metadata)

        return doc_nodes


async def _chunk_documents_with_metadata_async(
    docs: list[Document],
    splitter: SentenceSplitter,
    max_concurrent: int = OLLAMA_MAX_CONCURRENT_DOCS,
    show_progress: bool = True,
) -> list[BaseNode]:
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = [
        _process_document_async(doc, splitter, semaphore)
        for doc in docs
    ]

    if show_progress:
        from tqdm.asyncio import tqdm_asyncio

        results = await tqdm_asyncio.gather(
            *tasks,
            desc="Processing documents",
            total=len(docs),
            unit="doc",
        )
    else:
        results = await asyncio.gather(*tasks)

    all_nodes: list[BaseNode] = []
    for doc_nodes in results:
        all_nodes.extend(doc_nodes)
    return all_nodes


def chunk_documents_with_metadata(
    docs: list[Document],
    splitter: SentenceSplitter,
    show_progress: bool = True,
    max_concurrent: int = OLLAMA_MAX_CONCURRENT_DOCS,
) -> list[BaseNode]:
    return asyncio.run(
        _chunk_documents_with_metadata_async(
            docs,
            splitter,
            max_concurrent=max_concurrent,
            show_progress=show_progress,
        )
    )
