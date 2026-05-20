from pathlib import Path

from pypdf import PdfReader
import aiofiles


class DocumentLoader:
    async def load(self, file_path: str) -> str:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return await self._load_pdf(file_path)
        elif suffix in (".md", ".txt"):
            return await self._load_text(file_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식입니다: {suffix}")

    async def _load_pdf(self, file_path: str) -> str:
        reader = PdfReader(file_path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n".join(pages)

    async def _load_text(self, file_path: str) -> str:
        async with aiofiles.open(file_path, encoding="utf-8") as f:
            return await f.read()