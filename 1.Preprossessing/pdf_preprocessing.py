
#%%
import re
import json
import pdfplumber
import pandas as pd
from pathlib import Path
from collections import defaultdict
from pdfplumber.utils import extract_text
from typing import List


FUNCTION = "PDF PROCESSOR"

class PdfProcessor:
    def __init__(self, config: dict, image_analysis: bool = False):
        self.config = config
        self.image_analysis = image_analysis

    @staticmethod
    def parse_filename_metadata(pdf_path: str) -> dict:
        """
        Extract law_group, version_index, topic, and year from filename convention:
        {law_group}_{version_idx}_..._<Topic>_<year>.pdf
        e.g. 1_0_CELEX_02002L0044-20190726_EN_TXT_Vibration_2019.pdf
        """
        name = Path(pdf_path).stem
        match = re.match(r"^(\d+)_(\d+)_.*_([^_]+)_(\d{4})$", name)
        if match:
            return {
                "law_group": match.group(1),
                "version_index": int(match.group(2)),
                "topic": match.group(3),
                "year": int(match.group(4)),
            }
        return {"law_group": None, "version_index": None, "topic": None, "year": None}

    def process_text(self,
                     pdf_path: str,
                     project_root: str,
                    ):

        """
        Extract text from a Pdf file
        """

        # Output Folders
        pdf_name = Path(pdf_path).stem
        output_text_path = project_root / Path(self.config['folder']['txt_output']) / f"{pdf_name}.txt"
        output_text_path.parent.mkdir(parents=True, exist_ok=True)

        # Open file 
        pdf = pdfplumber.open(pdf_path)

        first_page = -1
        extracted_text = []
        with_tables_chars = []


        for idx, page in enumerate(pdf.pages):

            # Clean text
            cropped_chars = page.chars
            filtered_page=page

            # Search for tables
            try:
                raw_tables = filtered_page.find_tables()
                table_objs = filter_overlapping_tables(raw_tables)
            except Exception as e:
                raise e
 
            # Page without tables
            if not table_objs:
                try:
                    testo_unico = extract_text(filtered_page.chars, layout=True)
                    if testo_unico.strip():
                        page_text=testo_unico.strip()
                
                    page_text = re.sub(r'\n{6,}', '\n\n\n\n\n', page_text)
                except Exception as e:
                    raise e

            # Page with Tables
            else:
                tbl_list = []
                for tbl in table_objs:
                    try:
                        x0, y0, x1, y1 = tbl.bbox
                        tbl_list.append((y1, y0, tbl))
                        tbl_list.sort(key=lambda t: t[0], reverse=False)
                    except Exception as e:
                        raise e

                blocco_pag = []
                start=filtered_page.bbox[1]
                for i, (y1, y0, tbl) in enumerate(tbl_list, start=1):
                    try:
                        if y0 > start:
                            crop_region = filtered_page.within_bbox((filtered_page.bbox[0], start, filtered_page.width, y0)) 
                            testo_frag = extract_text(crop_region.chars, layout=True)  
                        
                            start=y1
                            if testo_frag.strip():
                                crop_text = re.sub(r'\n{6,}', '\n\n\n\n\n', testo_frag.strip())
                                blocco_pag.append("\n" + crop_text + "\n")

                        # Convert table in markdown
                        data = tbl.extract() 
                        if data:
                            headers_o = data[0]
                            rows = data[1:]
                            df = pd.DataFrame(rows, columns=headers_o)
                            md = df.to_markdown(index=False)
                            blocco_pag.append(md + "\n")
                    except Exception as e:
                        raise e

                try:
                    if filtered_page.bbox[3] > y1:
                        crop_region = filtered_page.within_bbox((0, y1, filtered_page.width, filtered_page.bbox[3]))
                        testo_basso = extract_text(crop_region.chars, layout=True) 
                        
                        if testo_basso.strip():
                            crop_text = re.sub(r'\n{6,}', '\n\n\n\n\n', testo_basso.strip())
                            blocco_pag.append("\n" + crop_text + "\n")
                except Exception as e:
                    raise e

                page_text="".join(blocco_pag)

            try:
                # Find first page
                pattern = r"^\s*1(?!\d) "
                pattern_2 = r"^\s*1(?!\d)."
                if (re.match(pattern, page_text) or re.match(pattern_2, page_text)) and first_page==-1: 
                    first_page = idx
            
                # Add page delimiter
                page_text = f"\n\n--- PAGE NUMBER: {idx+1} ---" + '\n\n'+ page_text
                #if self.image_analysis:
                    #page_text += "\n\n--- LLM IMAGE DESCRIPTION ---" + '\n\n'+image_text+ '\n\n'
                extracted_text.append(page_text)
                with_tables_chars.append(cropped_chars)
            except Exception as e:
                raise e    
            
        if first_page==-1:
            first_page=0

        
        extracted_text = extracted_text[first_page:]
        final_text=""
        for text in extracted_text:
            final_text += text
        with open(output_text_path, "w", encoding="utf-8") as f:
            f.write(final_text)
        
        return output_text_path#, index_path, is_index
    
    def from_text_to_json(self,
                          output_text_path: str,
                          project_root: str,
                          pdf_name: str):
        """
        Transform text in json, splitting for paragraph if index is present
        """

        metadata = self.parse_filename_metadata(pdf_name)

        with open(output_text_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Split documents by page
        try:
            raw_pages = re.split(r"--- PAGE NUMBER: (\d+) ---", text)
            pages = []
            for i in range(1, len(raw_pages), 2):
                pg = int(raw_pages[i])
                content = raw_pages[i+1]
                pages.append((pg, content))
        except Exception as e:
            raise e

        result = {}

        for page_num, page_content in pages:
            normal_text = page_content.strip()

            if self.image_analysis:
                # Extract LLM Block
                try:
                    llm_split = re.split(r"---\s*LLM IMAGE DESCRIPTION\s*---", page_content, flags=re.IGNORECASE)
                    normal_text = llm_split[0].strip()
                    for seg in llm_split[1:]:
                        llm_text = seg.strip()
                        if not llm_text:
                            continue
                        key = f"{page_num}_LLMContent"
                        result[key] = {
                            "content": llm_text,
                            "paragraph": None,
                            "page": [page_num],
                            "llm_image_description": True,
                            "doc_extension": "pdf",
                            **metadata,
                        }
                except Exception as e:
                    raise e

            # Page node
            try:
                key = f"{page_num}_TEXT"
                result[key] = {
                    "content": normal_text,
                    "paragraph": None,
                    "page": [page_num],
                    "llm_image_description": False,
                    "doc_extension": "pdf",
                    **metadata,
                }
            except Exception as e:
                raise e

        pdf_stem = Path(pdf_name).stem
        final_dict = {pdf_stem: result}
        json_path = project_root / Path(self.config['folder']['json_output']) / Path(pdf_stem) / f"json_{pdf_stem}.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(final_dict, f, ensure_ascii=False, indent=4)
        return json_path
# %%
def filter_overlapping_tables(table_objs: List["Table"]):
    """
    Dato un elenco di oggetti "table" di pdfplumber, restitusce solo quelli
    il cui bbox NON è contenuto all'interno di un altro bbox più grande.
    """

    tbl_info = []
    for idx, tbl in enumerate(table_objs):
        x0, y0, x1, y1 = tbl.bbox
        area = (x1 - x0) * (y1 - y0)
        tbl_info.append((area, idx, tbl.bbox, tbl))
    
    tbl_info.sort(key=lambda x: x[0], reverse=True)

    kept_tables = []
    kept_bboxes = []
    for area, idx, bbox, tbl in tbl_info:
        x0, y0, x1, y1 = bbox
        
        is_contained = False
        for kx0, ky0, kx1, ky1 in kept_bboxes:
            
            if x0 >= kx0 and x1 <= kx1 and y0 >= ky0 and y1 <= ky1:
                is_contained = True
                break
        if not is_contained:
            kept_tables.append(tbl)
            kept_bboxes.append(bbox)

    return kept_tables
