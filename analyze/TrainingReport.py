import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# Rejestracja czcionki obsługującej polskie znaki
try:
    pdfmetrics.registerFont(TTFont("DejaVuSans", "C:/Windows/Fonts/DejaVuSans.ttf"))
except:
    pdfmetrics.registerFont(TTFont("DejaVuSans", "arial.ttf"))


def save_training_report(output_path, context, all_metrics):
    doc = SimpleDocTemplate(output_path, pagesize=landscape(A4))
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Header", fontName="DejaVuSans", fontSize=18, leading=20, spaceAfter=12, alignment=1))
    styles.add(ParagraphStyle(name="SubHeader", fontName="DejaVuSans", fontSize=14, leading=16, spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="NormalUTF", fontName="DejaVuSans", fontSize=11, leading=13))
    story = []

    # === Nagłówek ===
    story.append(Paragraph("RAPORT KLASYFIKACJI UŻYTKOWANIA GRUNTÓW Z OBRAZÓW HIPERSPEKTRALNYCH", styles["Header"]))
    story.append(Paragraph(f"Data generacji: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["NormalUTF"]))
    story.append(Paragraph(f"Scena: {context.get('scene_name', '?')}", styles["NormalUTF"]))
    story.append(Paragraph(f"CRS: {context.get('crs', '?')} | Raster: {context.get('raster_shape', '?')}", styles["NormalUTF"]))
    story.append(Paragraph(f"PCA komponenty: {context.get('pca_k', '?')}", styles["NormalUTF"]))
    story.append(Spacer(1, 0.5 * cm))

    # === Tabela porównawcza ===
    story.append(Paragraph("Porównanie modeli", styles["SubHeader"]))
    table_data = [["Model", "Train Acc.", "Test Acc.", "Precision", "Recall", "F1", "Czas [s]"]]

    for metrics in all_metrics:
        name = metrics.get("name", "Model")
        train_acc = metrics.get("train", {}).get("accuracy", 0)
        test_acc = metrics.get("test", {}).get("accuracy", 0)
        test_report = metrics.get("test", {}).get("report_dict", {})
        precision = test_report.get("weighted avg", {}).get("precision", 0)
        recall = test_report.get("weighted avg", {}).get("recall", 0)
        f1 = test_report.get("weighted avg", {}).get("f1-score", 0)
        fit_time = metrics.get("timings", {}).get("fit_s", 0)

        table_data.append([
            name,
            f"{train_acc:.3f}",
            f"{test_acc:.3f}",
            f"{precision:.3f}",
            f"{recall:.3f}",
            f"{f1:.3f}",
            f"{fit_time:.2f}",
        ])

    table = Table(table_data, hAlign='LEFT', colWidths=[4*cm, 3*cm, 3*cm, 3*cm, 3*cm, 3*cm, 3*cm])
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "DejaVuSans"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#003366")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.black),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.8 * cm))

    # === Wykres dokładności modeli ===
    names = [m.get("name", "?") for m in all_metrics]
    train_accs = [m.get("train", {}).get("accuracy", 0) for m in all_metrics]
    test_accs = [m.get("test", {}).get("accuracy", 0) for m in all_metrics]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, train_accs, width, label="Train Acc.")
    ax.bar(x + width/2, test_accs, width, label="Test Acc.")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Porównanie dokładności modeli", fontsize=13, fontweight="bold")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
        fontsize=10
    )
    plt.subplots_adjust(top=0.9, bottom=0.3, left=0.1, right=0.95)
    chart_path = os.path.splitext(output_path)[0] + "_accuracy_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    story.append(Image(chart_path, width=15*cm, height=9*cm))
    story.append(Spacer(1, 1*cm))

    # === Krzywe uczenia (loss + acc) ===
    for metrics in all_metrics:
        model_name = metrics.get("name", "Unknown")
        curves = metrics.get("training_curves", None)
        if curves and "train_loss" in curves:
            epochs = np.arange(1, len(curves["train_loss"]) + 1)
            fig, ax1 = plt.subplots(figsize=(7, 5))

            # Loss (lewa oś)
            ax1.plot(epochs, curves["train_loss"], label="Train Loss", color="blue")
            ax1.plot(epochs, curves.get("val_loss", []), label="Val Loss", color="orange")
            ax1.set_xlabel("Epoka", fontsize=11)
            ax1.set_ylabel("Loss", fontsize=11)
            ax1.grid(True, linestyle="--", alpha=0.4)

            # Accuracy (prawa oś)
            if "train_acc" in curves:
                ax2 = ax1.twinx()
                ax2.plot(epochs, curves["train_acc"], label="Train Acc", color="green", linestyle="--")
                if "val_acc" in curves:
                    ax2.plot(epochs, curves["val_acc"], label="Val Acc", color="red", linestyle="--")
                ax2.set_ylabel("Accuracy", fontsize=11, color="green")

            ax1.set_title(f"Krzywe uczenia – {model_name}", fontsize=13, fontweight="bold")

            # 🔹 Legenda pod wykresem (z obu osi)
            lines, labels = ax1.get_legend_handles_labels()
            if "train_acc" in curves:
                lines2, labels2 = ax2.get_legend_handles_labels()
                lines += lines2
                labels += labels2

            ax1.legend(
                lines, labels,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.25),
                ncol=2,
                frameon=False,
                fontsize=9
            )
            plt.subplots_adjust(bottom=0.28, top=0.9, left=0.1, right=0.9)

            curve_path = os.path.splitext(output_path)[0] + f"_{model_name}_curves.png"
            plt.savefig(curve_path, dpi=150, bbox_inches="tight")
            plt.close()

            story.append(Image(curve_path, width=15*cm, height=9*cm))
            story.append(Spacer(1, 0.8*cm))

    # === Macierze pomyłek ===
    for metrics in all_metrics:
        model_name = metrics.get("name", "Unknown")
        cmatrix = np.array(metrics.get("test", {}).get("confusion_matrix", []))
        labels = metrics.get("test", {}).get("labels", [])
        if cmatrix.size > 0:
            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(cmatrix, cmap="Blues", interpolation="nearest")

            for i in range(cmatrix.shape[0]):
                for j in range(cmatrix.shape[1]):
                    val = cmatrix[i, j]
                    ax.text(j, i, str(val), ha="center", va="center",
                            color="black" if im.norm(val) < 0.6 else "white",
                            fontsize=11, fontweight="bold")

            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=10)
            ax.set_yticklabels(labels, fontsize=10)
            ax.set_xlabel("Predykcja", fontsize=11, labelpad=10)
            ax.set_ylabel("Prawda", fontsize=11, labelpad=10)
            ax.set_title(f"Macierz pomyłek – {model_name}", fontsize=13, fontweight="bold", pad=15)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.subplots_adjust(bottom=0.15, left=0.15, top=0.9, right=0.95)

            cm_path = os.path.splitext(output_path)[0] + f"_{model_name}_cm.png"
            plt.savefig(cm_path, dpi=200, bbox_inches="tight")
            plt.close()

            story.append(Image(cm_path, width=15*cm, height=11*cm))
            story.append(Spacer(1, 0.6*cm))

    doc.build(story)
    print(f"[DONE] Raport PDF zapisany: {output_path}")
