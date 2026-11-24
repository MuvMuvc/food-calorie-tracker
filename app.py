import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"

import re
import gradio as gr
import torch
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from datetime import datetime

import food_calories

MODEL_REPO = "XxxxxyKHML/food-calorie-resnet50"
MODEL_FILENAME = "resnet18.pth"
NUM_CLASSES = 101
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def load_model():
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    local_path = "./resnet18.pth"
    if os.path.exists(local_path):
        model_path = local_path
    else:
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


model = None
try:
    model = load_model()
except Exception:
    pass


def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return transform(image).unsqueeze(0)


def recognize(image):
    device = next(model.parameters()).device
    x = preprocess(image).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    idx = int(torch.argmax(probs).item())
    conf = float(probs[idx].item())
    name, cal = food_calories.FOODS[idx]
    return name, cal, conf


def today_table(history):
    today = datetime.now().strftime("%Y-%m-%d")
    rows = [[r["time"], r["name"], r["grams"], r["kcal"],
             '<span class="delete-icon" title="Delete">&#128465;</span>']
            for r in history if r["date"] == today]
    return pd.DataFrame(rows, columns=["Time", "Food", "Grams", "kcal", ""])


def render_summary(history, goal):
    today = datetime.now().strftime("%Y-%m-%d")
    total = sum(r["kcal"] for r in history if r["date"] == today)
    pct = total / goal * 100 if goal else 0
    bar_pct = min(pct, 100)

    if pct <= 70:
        color = "#4caf50"
    elif pct <= 100:
        color = "#ff9800"
    else:
        color = "#f44336"

    if goal < 1200:
        tip = "Below 1200 kcal may be too low for most adults."
    elif goal > 3500:
        tip = "Above 3500 kcal is quite high; please keep a balanced diet."
    else:
        tip = ""

    tip_html = (f'<div style="color:#e67e22;font-size:0.9em;margin-top:6px;">{tip}</div>'
                if tip else "")

    html = (
        '<div style="padding:4px 0;">'
        f'<div style="font-size:1.15em;font-weight:600;margin-bottom:6px;">Today: {total} / {int(goal)} kcal ({pct:.0f}%)</div>'
        '<div style="background:#eee;border-radius:12px;height:20px;width:100%;overflow:hidden;">'
        f'<div style="background:{color};width:{bar_pct:.1f}%;height:20px;"></div>'
        '</div>'
        f'{tip_html}'
        '</div>'
    )
    return html


def recognize_and_show(image):
    if image is None:
        return "### Upload a food photo to identify.", None
    if model is None:
        return "### Model is not loaded.", None
    name, cal100, conf = recognize(image)
    dn = food_calories.display_name(name)
    msg = (f"### Identified: **{dn}**\n\n"
           f"{cal100} kcal / 100g · confidence {conf * 100:.1f}%")
    return msg, name


def toggle_custom(on):
    return gr.update(visible=on), gr.update(visible=on)


def add_record(food_name, custom_on, custom_name, custom_cal, grams, goal, history):
    if custom_on:
        if not custom_name:
            return ("### Please enter a custom food name.", history,
                    today_table(history), render_summary(history, goal),
                    None, "### Upload a food photo to identify.", None,
                    False, None, None)
        try:
            cal100 = float(custom_cal)
        except Exception:
            cal100 = 0
        dn = str(custom_name).strip()
        kcal = round(cal100 * int(grams) / 100)
    else:
        if not food_name:
            return ("### Please upload and identify a food first.", history,
                    today_table(history), render_summary(history, goal),
                    None, "### Upload a food photo to identify.", None,
                    False, None, None)
        cal100 = None
        for n, c in food_calories.FOODS:
            if n == food_name:
                cal100 = c
                break
        if cal100 is None:
            return ("### Invalid food selection.", history,
                    today_table(history), render_summary(history, goal),
                    None, "### Upload a food photo to identify.", None,
                    False, None, None)
        dn = food_calories.display_name(food_name)
        kcal = round(cal100 * int(grams) / 100)

    now = datetime.now()
    record = {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M"),
        "name": dn,
        "grams": int(grams),
        "kcal": kcal,
    }
    new_history = history + [record]

    msg = f"### Recorded\n\n**{dn}** {int(grams)}g = **{kcal} kcal**"
    return (msg, new_history, today_table(new_history), render_summary(new_history, goal),
            None, "### Upload a food photo to identify.", None, False, None, None)


def on_select(history, goal, evt: gr.SelectData):
    if evt is None:
        return history, today_table(history), render_summary(history, goal)
    idx = evt.index
    if isinstance(idx, (tuple, list)):
        row = int(idx[0])
        col = int(idx[1]) if len(idx) > 1 else -1
    else:
        row = int(idx)
        col = -1
    if col != 4:
        return history, today_table(history), render_summary(history, goal)
    today = datetime.now().strftime("%Y-%m-%d")
    today_indices = [i for i, r in enumerate(history) if r["date"] == today]
    if 0 <= row < len(today_indices):
        del_idx = today_indices[row]
        new_history = [r for i, r in enumerate(history) if i != del_idx]
        return new_history, today_table(new_history), render_summary(new_history, goal)
    return history, today_table(history), render_summary(history, goal)


def is_valid_time(s):
    m = re.match(r"^(\d{1,2}):(\d{2})$", s)
    if not m:
        return False
    h = int(m.group(1))
    mi = int(m.group(2))
    return 0 <= h <= 23 and 0 <= mi <= 59


def match_food(name):
    s = str(name).strip()
    dn = s.title()
    if dn in DISPLAY_TO_CAL:
        return dn, DISPLAY_TO_CAL[dn]
    return s, None


def on_table_edit(history, goal, evt: gr.EditData):
    if evt is None:
        return history, today_table(history), render_summary(history, goal)
    idx = evt.index
    if isinstance(idx, (tuple, list)):
        row = int(idx[0])
        col = int(idx[1]) if len(idx) > 1 else -1
    else:
        row = int(idx)
        col = -1

    today = datetime.now().strftime("%Y-%m-%d")
    today_indices = [i for i, r in enumerate(history) if r["date"] == today]
    if not (0 <= row < len(today_indices)):
        return history, today_table(history), render_summary(history, goal)

    del_idx = today_indices[row]
    new_record = dict(history[del_idx])

    if col == 0:
        time_str = str(evt.value)
        if is_valid_time(time_str):
            new_record["time"] = time_str
    elif col == 1:
        name, cal100 = match_food(str(evt.value))
        new_record["name"] = name
        if cal100 is not None:
            new_record["kcal"] = round(cal100 * new_record["grams"] / 100)
    elif col == 2:
        grams = int(evt.value)
        new_record["grams"] = grams
        cal100 = DISPLAY_TO_CAL.get(new_record["name"])
        if cal100 is not None:
            new_record["kcal"] = round(cal100 * grams / 100)
    elif col == 3:
        kcal = int(evt.value)
        cal100 = DISPLAY_TO_CAL.get(new_record["name"])
        if cal100 is not None:
            new_record["grams"] = round(kcal * 100 / cal100)
        new_record["kcal"] = kcal

    new_history = [r for i, r in enumerate(history) if i != del_idx]
    new_history.insert(del_idx, new_record)
    return new_history, today_table(new_history), render_summary(new_history, goal)


def search_food(query):
    if not query:
        return ""
    q = query.strip().lower()
    results = []
    for n, c in food_calories.FOODS:
        dn = food_calories.display_name(n)
        if q in dn.lower() or q in n.lower():
            results.append(f"- **{dn}**: {c} kcal / 100g")
    if not results:
        return "### No matching food found."
    if len(results) > 20:
        results = results[:20] + ["- ..."]
    return "### Results\n\n" + "\n".join(results)


def update_goal(goal, history):
    return render_summary(history, goal)


def clear_today(history, goal):
    today = datetime.now().strftime("%Y-%m-%d")
    new_history = [r for r in history if r["date"] != today]
    return new_history, today_table(new_history), render_summary(new_history, goal)


FOOD_CHOICES = [(food_calories.display_name(n), n) for n, _ in food_calories.FOODS]
DISPLAY_TO_CAL = {food_calories.display_name(n): c for n, c in food_calories.FOODS}
CAL_TABLE = pd.DataFrame([(food_calories.display_name(n), c) for n, c in food_calories.FOODS],
                         columns=["Food", "kcal/100g"])

CSS = """
.gradio-container {
    max-width: 1150px !important;
    margin: 0 auto !important;
}
#food_image {
    height: 320px !important;
    max-height: 320px !important;
}
.delete-icon {
    opacity: 0.35;
    cursor: pointer;
    transition: none !important;
}
.delete-icon:hover {
    opacity: 1;
}
footer {
    display: none !important;
}
"""

theme = gr.themes.Soft(primary_hue="orange", secondary_hue="amber", neutral_hue="stone")

with gr.Blocks(title="Food Calorie Tracker") as demo:
    gr.HTML(
        """
        <div style="text-align:center; padding:16px 0 22px 0;">
            <h1 style="font-size:2.4em; margin:0 0 8px 0;">Food Recognition &amp; Calorie Tracker</h1>
            <p style="font-size:1.15em; color:#8a8a8a; margin:0;">Take a photo of your food and track your daily calories.</p>
        </div>
        """
    )

    history_state = gr.State([])

    with gr.Tabs():
        with gr.Tab("Log a Meal", render_children=True):
            with gr.Row():
                image_input = gr.Image(
                    type="pil",
                    label="Food photo (upload or camera)",
                    sources=["upload", "webcam"],
                    height=320,
                    elem_id="food_image",
                )
                with gr.Column():
                    recognized_msg = gr.Markdown("### Upload a food photo to identify.")
                    food_dropdown = gr.Dropdown(choices=FOOD_CHOICES,
                                                label="Confirm or change food", value=None)
                    custom_checkbox = gr.Checkbox(label="Custom food (not in the list)", value=False)
                    custom_name = gr.Textbox(label="Custom food name", visible=False)
                    custom_cal = gr.Number(label="Custom kcal / 100g", visible=False)
                    grams_slider = gr.Slider(50, 1000, value=150, step=10,
                                             label="Amount (grams)")
                    record_btn = gr.Button("Log this meal", variant="primary", size="lg")
                    record_msg = gr.Markdown()

        with gr.Tab("Today", render_children=True):
            goal_slider = gr.Slider(800, 5000, value=2000, step=50,
                                    label="Daily calorie goal (kcal)")
            today_summary = gr.HTML(render_summary([], 2000))
            today_table_out = gr.Dataframe(
                value=today_table([]), label="Today's log",
                wrap=True, interactive=True,
                datatype=["str", "str", "number", "number", "html"],
                column_widths=["15%", "50%", "15%", "15%", "5%"],
            )
            clear_btn = gr.Button("Clear today", size="sm")

        with gr.Tab("Calorie Reference", render_children=True):
            search_box = gr.Textbox(label="Search a food", placeholder="e.g. pizza or cake")
            search_result = gr.Markdown()
            gr.Markdown("Calorie reference for 101 food categories (per 100g).")
            gr.Dataframe(value=CAL_TABLE, label="Food calorie table", interactive=True, wrap=True,
                         column_widths=["70%", "30%"])

    image_input.change(recognize_and_show, inputs=image_input, outputs=[recognized_msg, food_dropdown])
    custom_checkbox.change(toggle_custom, inputs=custom_checkbox, outputs=[custom_name, custom_cal])
    record_btn.click(
        add_record,
        inputs=[food_dropdown, custom_checkbox, custom_name, custom_cal,
                grams_slider, goal_slider, history_state],
        outputs=[record_msg, history_state, today_table_out, today_summary,
                 image_input, recognized_msg, food_dropdown,
                 custom_checkbox, custom_name, custom_cal],
    )
    today_table_out.select(
        on_select,
        inputs=[history_state, goal_slider],
        outputs=[history_state, today_table_out, today_summary],
    )
    today_table_out.edit(
        on_table_edit,
        inputs=[history_state, goal_slider],
        outputs=[history_state, today_table_out, today_summary],
    )
    search_box.change(search_food, inputs=search_box, outputs=search_result)
    goal_slider.change(update_goal, inputs=[goal_slider, history_state], outputs=[today_summary])
    clear_btn.click(
        clear_today,
        inputs=[history_state, goal_slider],
        outputs=[history_state, today_table_out, today_summary],
    )

    gr.HTML(
        """
        <div style="text-align:center; color:#b0b0b0; font-size:0.9em; padding:14px 0 4px 0;">
            For learning/demo purposes only. Calorie values are approximate.
        </div>
        """
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), theme=theme, css=CSS)
