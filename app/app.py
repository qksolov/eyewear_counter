import gradio as gr
from eyewear_counter import EyewearCounter, generate_report

import tempfile
import os
import pandas as pd
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = EyewearCounter(device=device)

example_file_path = "https://github.com/qksolov/eyewear-counter/raw/refs/heads/main/assets/example_data.csv"
example_df = pd.read_csv(example_file_path, encoding='cp1251', sep=';')
_, summary_df_dummy, plot_df_dummy = generate_report([], 0)
summary_df_dummy.replace(0, "", inplace=True)

column_placeholder = "Выберете столбец с URL изображений"


class GradioProgressBar:
    """Обертка над gr.Progress для отображения прогресса в интерфейсе во время работы EyewearCounter."""
    def __init__(self, total, desc="", progress=gr.Progress(), **kwargs):
        self.total = total
        self.desc = desc
        self.current = 0
        self.progress = progress
        self.progress(0, desc=f"{self.desc}: 0/{self.total}")

    def update(self, n=1):
        self.current += n
        self.progress(self.current / self.total, desc=f"{self.desc}: {self.current}/{self.total}")

    def set_description(self, desc):
        self.desc = desc
        self.progress(self.current / self.total, desc=self.desc)
    
    def write(self, text):
        pass    

    def close(self):
        pass


def upload_file(file=None):
    try:
        if file is None:
            df = example_df.copy()
            temp_dir = tempfile.gettempdir()
            file = os.path.join(temp_dir, "example_data.csv")
            df.to_csv(file, sep=";", index=False, encoding="utf-8-sig")
            column = 'image_url'
            column_elem_id = None
            btn_variant = 'primary'
            btn_interactive = True
        else:
            ext = os.path.splitext(file)[-1].lower()
            if ext == '.csv':
                df = pd.read_csv(file, sep=';', encoding='cp1251')
            elif ext in ('.xls', '.xlsx', '.xlsm', '.xlsb'):
                df = pd.read_excel(file, sheet_name=0)
            else:
                gr.Warning(
                    "Неподдерживаемый формат файла. Допустимы: .csv, .xls, .xlsx, .xlsm, .xlsb.",
                    title="Error"
                )
                return delete_file()
            column = column_placeholder
            column_elem_id = "active"
            btn_variant = 'secondary'
            btn_interactive = False
    except Exception:
        gr.Warning(f"Не удалось загрузить файл {file}", title="Error")
        return delete_file()

    columns = [column_placeholder] + list(df.columns)
    return (
        gr.File(value=None, visible=False),                                     # file_upload
        gr.File(file, visible=True, interactive=True),                          # file_input
        df,                                                                     # df_input_state
        gr.Dataframe(df, visible=True),                                         # df_show
        gr.Dropdown(value=column, choices=columns, elem_id=column_elem_id),     # url_column
        gr.Button(interactive=btn_interactive, variant=btn_variant),            # run_button
    )


def delete_file():
    return (
        gr.File(value=None, visible=True),                                      # file_upload
        gr.File(interactive=False),                                             # file_input
        None,                                                                   # df_input_state
        gr.Dataframe(value=None, visible=False),                                # df_show
        gr.Dropdown(                                                            # url_column
            value=column_placeholder, choices=[column_placeholder],elem_id=None),
        gr.Button(interactive=False, variant='secondary'),                      # run_button
    )


def select_column(url_column):    
    if url_column == column_placeholder:
        column_elem_id = "active"
        btn_variant = 'secondary'
        btn_interactive = False
    else:
        column_elem_id = None
        btn_variant = 'primary'
        btn_interactive = True
    return (
        gr.Dropdown(elem_id=column_elem_id),                                    # url_column
        gr.Button(interactive=btn_interactive, variant=btn_variant)             # run_button
    )


def run_model(
        df, url_column,
        image_size, image_fit, max_faces, threshold, max_workers, batch_size
        ):
    image_urls = list(df[url_column])
    try:
        predictions_tensor, errors_cnt = model.run(
            image_urls,
            image_size=image_size, image_fit=image_fit,
            max_faces=max_faces, threshold=threshold,
            max_workers=max_workers, batch_size=batch_size,
            progress_bar=GradioProgressBar,
            save_samples=True
        )
        if len(image_urls) == errors_cnt:
            gr.Warning(f"Не удалось загрузить изображения. Проверте столбей с URL.", title="Error")
            return df, None, summary_df_dummy, plot_df_dummy, None, None, None
    except Exception as e:
        gr.Warning(f"Ошибка обработки: {e}", title="Error")
        return df, None, summary_df_dummy, plot_df_dummy, None, None, None
    
    predictions = predictions_tensor.cpu().numpy()

    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, "report.xlsx")
    df_report, summary_df, plot_df = generate_report(
        predictions, errors_cnt, df_input=df, filename=file_path)
    
    image_samples = model.samples
    img0 = image_samples[0] if 0 in image_samples else None
    img1 = image_samples[1] if 1 in image_samples else None
    img2 = image_samples[2] if 2 in image_samples else None

            # df_show,  file_output,     summary, barplot, img0, img1, img2
    return df_report, gr.File(file_path), summary_df, plot_df, img0, img1, img2


css = """
#interface-container {
    max-width: 1300px !important;
    margin: 0 auto !important;
}
#file-container {
    height: 63px !important;
}
#active {
    outline: 2px solid #3b82f6 !important;
    outline-offset: -1px !important;
}
#btn {
    font-weight: 400 !important;
    font-size: 15px !important;
    text-shadow: 0.75px 0 0 currentColor;
    height: 41px !important;
}
.sort-buttons {
    display: none !important;
}
"""


with gr.Blocks(css=css, theme=gr.themes.Base()) as demo:
    df_input_state = gr.State()

    with gr.Row(elem_id="interface-container", scale=0, equal_height=True):
        gr.Markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <div style="display: flex; align-items: baseline; gap: 20px;">
                    <h1 style="margin: 0;">Eyewear Counter</h1>
                    <a href="https://github.com/qksolov/eyewear-counter" target="_blank" style="color: #3b82f6; font-size: 14px">
                        GitHub
                    </a>
                </div>
                <span style="color: #97979e; font-size: 14px;">Устройство: {"[GPU]" if device == 'cuda' else "[CPU]"}</span>
            </div>
            """,
            sanitize_html=False
        )


    with gr.Row(elem_id="interface-container", scale=0):
        with gr.Column(scale=1):
            file_input = gr.File(
                label='Файл для анализа', 
                interactive=False, elem_id="file-container")

            url_column = gr.Dropdown(
                value=column_placeholder,
                choices=[column_placeholder],
                label="Выберете столбец с URL изображений",
                container=False, elem_id="inactive"
            )
            run_button = gr.Button(
                "Запустить обработку",
                size="lg", variant='secondary',
                interactive=False, elem_id="btn"
            )

            with gr.Group():
                with gr.Accordion("Инструкция", open=True) as acc1:
                    gr.Radio(label="""
                             Поддерживаемые форматы файлов: .csv, .xlx, .xlsx, .xlsm, .xlsb.
                             Если в файле несколько листов, будет загружен только первый.
                             Для .csv файлов разделителем должна быть точка с запятой (;).
                             """)
                    gr.Radio(label="""
                             В таблице должен быть столбец с URL-ссылками на изображения 
                             для анализа.
                             """)
                    gr.Radio(label="""
                             Для ускоренной обработки необходимо наличие доступа к [GPU].
                             """)
                    gr.Radio(label="""
                             Модель детектит лица на изображении и определяет их класс.
                             В результате выводится количество лиц каждого класса.
                             """)
                
                with gr.Accordion("Продвинутые настройки", open=False) as acc2:
                    image_size = gr.Radio(
                        value=640, choices=[1024, 640, 512],
                        label="Размер обрабатываемых изображений",
                    )
                    image_fit = gr.Radio(
                        value="pad", choices=["pad", "stretch"],
                        label="Режим масштабирования"
                    )
                    max_faces = gr.Slider(
                        value=4, minimum=1, maximum=10, step=1,
                        label="Лимит распозноваемых лиц",
                        show_reset_button=False,
                    )
                    threshold = gr.Slider(
                        value=0.7, minimum=0.5, maximum=0.95, step=0.05,
                        label="Порог уверенности лица",
                        show_reset_button=False,
                    )
                    batch_size = gr.Slider(
                        value=32 if device == 'cuda' else 8, minimum=1, maximum=100, step=1,
                        label="Размер батча",
                        show_reset_button=False,
                    )
                    max_workers = gr.Radio(                        
                        value=2, choices=[1, 2, 3, 4],
                        label="Максимальное количество потоков",
                    )
            
            acc1.expand(lambda: gr.update(open=False), outputs=acc2)
            acc2.expand(lambda: gr.update(open=False), outputs=acc1)

            text_dummy = gr.TextArea(visible=False)
            example = gr.Dataset(
                label='Пример данных', samples=[["example_data.csv"]],
                components=[text_dummy]
            )

        with gr.Column(scale=3, min_width=400):
            file_upload = gr.File(
                label='Загрузите файл для анализа',
                elem_id="active", height=298, interactive=True,
                file_count='single'
            )
            df_show = gr.Dataframe(max_height=300, column_widths=['10%']*10, visible=False)
            
            file_output = gr.File(
                label='Отчет',
                elem_id="file-container", interactive=False, visible=True)

            with gr.Row(equal_height=True):
                with gr.Column():
                    summary = gr.Dataframe(
                        summary_df_dummy,
                        interactive=False, max_height=345,
                        column_widths=['20%', '10%', '8%'],
                        wrap=True, show_search=None, min_width=400
                    )
                with gr.Column(min_width=400):
                    with gr.Row(equal_height=True, height=345):
                        barplot = gr.BarPlot(
                            plot_df_dummy, height=345,
                            x="Категория", y="Количество", x_title=" ",
                            color="Тип",
                            color_map={"Без других классов":"#3b82f6d1", 
                                       "Есть другие классы":"#275aacd1"},
                            sort=['Очки', 'Без', 'Солнце']
                        )
                        with gr.Column(min_width=100, scale=0):
                            img0 = gr.Image(
                                label='Очки', interactive=False, 
                                min_width=100, height=100,                                
                                show_download_button=False, show_fullscreen_button=False
                            )
                            img1 = gr.Image(
                                label='Без', interactive=False, 
                                min_width=100, height=100,  
                                show_download_button=False, show_fullscreen_button=False
                            )
                            img2 = gr.Image(
                                label='Солнце', interactive=False, 
                                min_width=100, height=100, 
                                show_download_button=False, show_fullscreen_button=False
                            )


    file_upload.upload(
        upload_file, 
        file_upload,
        [file_upload, file_input, df_input_state, df_show, url_column, run_button]
    )
    
    example.click(
        upload_file,
        None,
        [file_upload, file_input, df_input_state, df_show, url_column, run_button]
    )

    file_input.clear(
        delete_file,
        outputs=[file_upload, file_input, df_input_state, df_show, url_column, run_button]
    )

    url_column.select(select_column, url_column, [url_column, run_button])

    run_button.click(
        run_model,
        [df_input_state, url_column,
            image_size, image_fit, max_faces, threshold, max_workers, batch_size],
        [df_show, file_output, summary, barplot, img0, img1, img2],
        show_progress_on=df_show
    )


def main():
    demo.launch(share=True)

if __name__ == "__main__":
    main()