# import pymupdf4llm
# from langchain_text_splitters import MarkdownHeaderTextSplitter
# from utils import loadllm
# import os

# # os.environ["OLLAMA_BASE_URL"] = "http://10.129.152.197:11434"
# os.environ["OLLAMA_BASE_URL"] = "http://10.195.100.5:11434"


# class themegen:
#     def __init__(self, llm, pdf_path):
#         self.llm = llm
#         self.pdf_path = pdf_path

#     def convert_pdf_to_markdown(self):
#         return pymupdf4llm.to_markdown(self.pdf_path)

#     def extract_theme_info(self, md_text) -> (list, list):
#         headers_to_split_on = [
#             ("#", "heading"),
#         ]

#         markdown_splitter = MarkdownHeaderTextSplitter(
#             headers_to_split_on=headers_to_split_on
#         )
#         md_header_splits = markdown_splitter.split_text(md_text)
#         start = 0
#         themesummary = []
#         themeinformation = []
#         for i in range(len(md_header_splits)):
#             if (
#                 "welcome to" in md_header_splits[i].metadata["heading"].lower()
#                 or "introduction" in md_header_splits[i].metadata["heading"].lower()
#                 or "theme description"
#                 in md_header_splits[i].metadata["heading"].lower()
#             ):
#                 themesummary.append(md_header_splits[i].page_content)

#             if (
#                 "task" in md_header_splits[i].metadata["heading"].lower()
#                 or i == len(md_header_splits) - 1
#             ):
#                 temp = f"\n Heading: {md_header_splits[start].metadata['heading']} \n "
#                 for j in range(start, i):
#                     temp += f" \n {md_header_splits[j].metadata['heading']} - Content: {md_header_splits[j].page_content}"
#                 if start == 0:
#                     crap = temp
#                 else:
#                     themeinformation.append(temp)
#                 start = i
#         with open("themeinformation.txt", "w", encoding="utf-8") as f:
#             f.write("\n".join(themeinformation))
#         return themesummary, themeinformation

#     def transform_theme_summary_and_introduction(self, themeinfo):
#         prompt = f"Generate a similar theme based on the given theme: {themeinfo}"
#         return self.llm.complete(prompt)

#     def generate_tasks(self, new_theme_info, tasks):
#         i = 0
#         prompt = f"""Generate detailed task 0 based on the Theme taking 
#         inspiration from the Example task but be different and creative from it.
#         \n
#         ======================================================================================
#         Theme: {new_theme_info}
#         ======================================================================================
#         \n
#         ======================================================================================
#         Example task: {tasks[i]}
#         ======================================================================================
#         """
#         print("PROMPT LENGTH: ", len(prompt))
#         print(self.llm.complete(prompt))
#         while True:
#             input_text = input(
#                 "(y): Generate next task \n(r): Regenerate task \n(n): Exit\n"
#             )
#             if input_text.lower() == "y":
#                 i += 1
#                 prompt = f"""Generate detailed task {i} based on the Theme and the Previous task. 
#                 Take inspiration from the Example task but be different and creative from it.
#                 \n
#                 ======================================================================================
#                 Theme: {new_theme_info}
#                 ======================================================================================
#                 \n
#                 ======================================================================================
#                 Previous Task / Task {i-1}: {tasks[i-1]}
#                 ======================================================================================
#                 \n
#                 ======================================================================================
#                 Example task: {tasks[i]}
#                 ======================================================================================
#                 """
#                 print("PROMPT LENGTH: ", len(prompt))
#                 print(self.llm.complete(prompt))
#             elif input_text.lower() == "r":
#                 user_prompt = input("Please enter the new prompt:")

#                 prompt = f"""Generate the next detailed task {i} based on the Theme and the Previous task. 
#                 Take inspiration from the Example task but be different and creative from it.
#                 \n
#                 FOLLOW THESE INSTRUCTIONS PLEASE: {user_prompt}
#                 ======================================================================================
#                 Theme: {new_theme_info}
#                 ======================================================================================
#                 \n
#                 ======================================================================================
#                 Previous Task / Task {i-1}: {tasks[i-1]}
#                 ======================================================================================
#                 \n
#                 ======================================================================================
#                 Example task: {tasks[i]}
#                 ======================================================================================
#                 """
#                 print("PROMPT LENGTH: ", len(prompt))
#                 print(self.llm.complete(prompt))
#             else:
#                 break

#     def run(self):
#         markdown = self.convert_pdf_to_markdown()
#         theme_info, old_tasks = self.extract_theme_info(markdown)
#         new_theme_info = self.transform_theme_summary_and_introduction(
#             "\n".join(theme_info)
#         )
#         print(new_theme_info)
#         print(
#             "--------------------------------------------------------------------------------------"
#         )
#         self.generate_tasks(new_theme_info, old_tasks)


# themegen(
#     llm=loadllm("Ollama", name="mixtral:8x7b"),
#     # llm=loadllm("Ollama"),
#     pdf_path="data/themebooks/GG23.pdf",
# ).run()
