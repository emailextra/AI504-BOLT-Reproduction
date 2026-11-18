"""Render the research report and its figures from verified evidence tables."""
from pathlib import Path
import csv
import html
import json
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

ROOT=Path(__file__).resolve().parents[1]


def read(name):
    """Load a generated report table."""
    with (ROOT/"results/tables"/f"{name}.csv").open() as handle:
        return list(csv.DictReader(handle))


def figures():
    """Plot fold variability and the paired scores without selecting a winning detector."""
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,
                         "axes.spines.right":False,"axes.labelcolor":"#26354a"})
    rows=read("deepunk_folds")
    refs=read("deepunk_comparison")
    fig,axes=plt.subplots(1,2,figsize=(7,2.5),sharey=True)
    for ax,metric,title in zip(axes,["K-F1","N-F1"],["Known class F1","Unknown class F1"]):
        for x,ratio in enumerate(["0.25","0.5"]):
            values=[float(row[metric]) for row in rows if row["known_ratio"]==ratio]
            ax.scatter([x+(i-2)*.035 for i in range(5)],values,color="#245c87",s=22,zorder=3)
            ref=next(row for row in refs if row["known_ratio"]==ratio and row["metric"]==metric)
            ax.scatter(x,float(ref["mean"]),marker="D",color="#245c87",s=36,label="Reproduced mean" if x==0 else None)
            ax.scatter(x,float(ref["published"]),marker="_",color="#bd642d",s=260,linewidths=2,label="Published" if x==0 else None)
        ax.set_xticks([0,1],["25%","50%"])
        ax.set_xlabel("Known intent categories")
        ax.set_title(title,loc="left",fontweight="bold",fontsize=10)
        ax.grid(axis="y",color="#e6e9ee")
        ax.set_ylim(0,65)
    axes[0].set_ylabel("F1 (%)")
    axes[1].legend(frameon=False,loc="upper left",fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT/"report/figure1.png",dpi=220,bbox_inches="tight")
    plt.close(fig)
    rows=read("knncon_paired")
    fig,axes=plt.subplots(1,2,figsize=(7,2.5),sharey=True)
    for ax,metric,title in zip(axes,["K-F1","N-F1"],["Known class F1","Unknown class F1"]):
        for offset,variant,label,color in [(-.18,"full","Released mask","#245c87"),(.18,"mask","Corrected mask","#bd642d")]:
            values=[float(next(r for r in rows if r["variant"]==variant and r["detector"]==d)[metric])
                    for d in ["lof_cosine","lof_euclidean"]]
            bars=ax.bar([offset,1+offset],values,.34,label=label,color=color)
            ax.bar_label(bars,fmt="%.1f",padding=3,fontsize=8)
        ax.set_xticks([0,1],["Cosine LOF","Euclidean LOF"])
        ax.set_title(title,loc="left",fontweight="bold",fontsize=10)
        ax.set_ylim(0,100)
        ax.grid(axis="y",color="#e6e9ee")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("F1 (%)")
    axes[1].legend(frameon=False,fontsize=7,loc="upper right")
    fig.tight_layout()
    fig.savefig(ROOT/"report/figure2.png",dpi=220,bbox_inches="tight")
    plt.close(fig)


def render():
    """Build a readable research-paper PDF with live text and embedded fonts."""
    font_dir=Path("C:/Windows/Fonts")
    if font_dir.exists():
        for name,filename in [("Research","times.ttf"),("Research-Bold","timesbd.ttf"),("Research-Italic","timesi.ttf")]:
            pdfmetrics.registerFont(TTFont(name,str(font_dir/filename)))
        pdfmetrics.registerFontFamily("Research",normal="Research",bold="Research-Bold",italic="Research-Italic",boldItalic="Research-Bold")
    else:
        # Standard PDF Times faces are portable when the Windows fonts are unavailable.
        pdfmetrics.registerFontFamily("Research",normal="Times-Roman",bold="Times-Bold",italic="Times-Italic",boldItalic="Times-Bold")
    normal="Research" if font_dir.exists() else "Times-Roman"
    bold="Research-Bold" if font_dir.exists() else "Times-Bold"
    styles={
      "body":ParagraphStyle("body",fontName=normal,fontSize=10.5,leading=14,spaceAfter=7,alignment=TA_JUSTIFY),
      "title":ParagraphStyle("title",fontName=bold,fontSize=20,leading=23,spaceAfter=12),
      "subtitle":ParagraphStyle("subtitle",fontName=normal,fontSize=10,leading=13,spaceAfter=12,textColor=colors.HexColor("#465269")),
      "h1":ParagraphStyle("h1",fontName=bold,fontSize=13,leading=16,spaceBefore=12,spaceAfter=7,keepWithNext=True),
      "h2":ParagraphStyle("h2",fontName=bold,fontSize=11.5,leading=14,spaceBefore=9,spaceAfter=5,keepWithNext=True),
      "caption":ParagraphStyle("caption",fontName=normal,fontSize=9,leading=11.5,spaceAfter=10),
      "table":ParagraphStyle("table",fontName=normal,fontSize=8.7,leading=11),
      "tablehead":ParagraphStyle("tablehead",fontName=bold,fontSize=8.7,leading=11,textColor=colors.white),
      "reference":ParagraphStyle("reference",fontName=normal,fontSize=9.5,leading=12,spaceAfter=6)
    }
    width=A4[0]-44*mm
    def para(text,kind="body"):
        text=html.escape(text)
        text=re.sub(r"https://[^ ]+",lambda m:f'<link href="{m.group()}">{m.group()}</link>',text)
        return Paragraph(text,styles[kind])
    def table(headers,rows,widths):
        data=[[para(str(x),"tablehead") for x in headers]]
        data.extend([[para(str(x),"table") for x in row] for row in rows])
        t=Table(data,colWidths=[width*x for x in widths],repeatRows=1,hAlign="LEFT")
        t.setStyle(TableStyle([
          ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#263c55")),
          ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f1f4f7"),colors.white]),
          ("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),7),
          ("RIGHTPADDING",(0,0),(-1,-1),7),("TOPPADDING",(0,0),(-1,-1),6),
          ("BOTTOMPADDING",(0,0),(-1,-1),6),
          ("LINEBELOW",(0,-1),(-1,-1),.5,colors.HexColor("#bbc5d0"))]))
        return t
    comparisons=read("deepunk_comparison")
    table1=[]
    for row in comparisons:
        table1.append([f'{float(row["known_ratio"])*100:.0f}%',
                       "Known F1" if row["metric"]=="K-F1" else "Unknown F1",
                       f'{float(row["published"]):.2f}',f'{float(row["mean"]):.2f} ± {float(row["fold_sd"]):.2f}',
                       f'{float(row["delta"]):+.2f}'])
    paired=read("knncon_paired")
    table2=[]
    for row in paired:
        table2.append(["Released" if row["variant"]=="full" else "Corrected",
                       "Cosine" if row["detector"]=="lof_cosine" else "Euclidean",
                       *[f'{float(row[k]):.2f}' for k in ("K-F1","N-F1","ACC")]])
    errors=read("errors")
    table3=[]
    for mode in ("full","mask"):
        for detector in ("lof_cosine","lof_euclidean"):
            match=[r for r in errors if r["trial"]==f"short_5epoch/{mode}/knncon_0.25_0" and r["detector"]==detector]
            counts={r["category"]:r["count"] for r in match}
            table3.append(["Released" if mode=="full" else "Corrected",
                           "Cosine" if detector=="lof_cosine" else "Euclidean",
                           *[counts[x] for x in ("correct","unknown accepted","known rejected","wrong known class")]])
    blocks={
      "[TABLE_DEEPUNK]":[para("Table 1. DeepUNK means and sample standard deviations across five class partitions. All scores and differences are percentage points. Published values are from BOLT Tables S4 and S7.","caption"),
                           table(["Known classes","Metric","Published","Reproduced ± SD","Difference"],table1,[.16,.19,.16,.32,.17]),Spacer(1,8)],
      "[TABLE_KNNCON]":[para("Table 2. Single-partition KNNCon mask comparison after five epochs. Both distance variants are retained. Scores are percentages.","caption"),
                           table(["Mask","LOF distance","Known F1","Unknown F1","Accuracy"],table2,[.19,.23,.18,.22,.18]),Spacer(1,8)],
      "[TABLE_ERRORS]":[para("Table 3. Error counts over 3,080 requests for each paired KNNCon variant. Wrong known means confusion between familiar intents.","caption"),
                        table(["Mask","Distance","Correct","Unknown accepted","Known rejected","Wrong known"],table3,[.17,.17,.13,.19,.18,.16]),Spacer(1,8)]
    }
    for marker,filename,caption in [
      ("[FIGURE_DEEPUNK]","figure1.png","Figure 1. DeepUNK fold scores (dots), means (diamonds), and published values (orange marks). The wide spread in unknown-class F1 at 50% known classes is visible."),
      ("[FIGURE_MASK]","figure2.png","Figure 2. Released and corrected KNNCon scores with cosine and Euclidean outlier detection. Both use one partition, one seed, and five epochs.")]:
        blocks[marker]=[Image(str(ROOT/"report"/filename),width=width,height=width*2.5/7),para(caption,"caption")]
    details=json.loads((ROOT/"submission-details.json").read_text())
    members=details.get("members",[])
    contribution=[]
    for member in members:
        contribution.append([member.get("name",""),member.get("id",""),member.get("responsibilities",""),
                             member.get("sections",""),member.get("code",""),str(member.get("percent","")),
                             member.get("agreed_name","")])
    if not contribution or sum(member["percent"] for member in members) != 100:
        raise ValueError("A complete agreed contribution allocation totaling 100% is required")
    blocks["[TABLE_CONTRIBUTION]"]=[table(["Name","Student ID","Responsibilities","Report sections","Code files","Share %","Agreed name"],contribution,[.14,.13,.19,.10,.20,.08,.16])]
    story=[]
    text=(ROOT/"report/Report.md").read_text(encoding="utf-8-sig")
    paragraphs=re.split(r"\n\s*\n",text)
    for block in paragraphs:
        block=block.strip()
        if not block: continue
        if block in blocks:
            story.append(KeepTogether(blocks[block]))
        elif block.startswith("# "):
            story.append(para(block[2:],"title"))
        elif block.startswith("### "):
            story.append(para(block[4:],"h2"))
        elif block.startswith("## "):
            if block=="## Contribution statement": story.append(PageBreak())
            story.append(para(block[3:],"h1"))
        elif block.startswith("AI504 "):
            story.append(Paragraph("<br/>".join(html.escape(line.strip()) for line in block.splitlines() if line.strip()), styles["subtitle"]))
            if members:
                story.append(para("Group " + str(int(details["group_number"])), "subtitle"))
                for start in range(0, len(members), 2):
                    identity = " · ".join(member["name"] + " (" + member["id"] + ")" for member in members[start:start + 2])
                    story.append(para(identity, "subtitle"))
            if details.get("repository_url"):
                story.append(para("Repository " + details["repository_url"], "subtitle"))
        else:
            kind="reference" if re.match(r"^\[\d+\]",block) else "body"
            story.append(para(block,kind))
    def page(canvas,doc):
        canvas.saveState()
        canvas.setFont(normal,8)
        canvas.setFillColor(colors.HexColor("#586778"))
        if doc.page>1:
            canvas.drawString(22*mm,A4[1]-14*mm,"AI504  |  BOLT reproduction on BANKING77")
            canvas.setStrokeColor(colors.HexColor("#d5dce3"))
            canvas.line(22*mm,A4[1]-16*mm,A4[0]-22*mm,A4[1]-16*mm)
        canvas.drawString(22*mm,13*mm,"Research reproduction study")
        canvas.drawRightString(A4[0]-22*mm,13*mm,str(doc.page))
        canvas.restoreState()
    doc=SimpleDocTemplate(str(ROOT/"Report.pdf"),pagesize=A4,leftMargin=22*mm,rightMargin=22*mm,
                          topMargin=23*mm,bottomMargin=22*mm,
                          title="Reproducing BOLT under Limited Supervision for Banking Intents",
                          author=", ".join(member["name"] for member in members),pageCompression=1)
    doc.build(story,onFirstPage=page,onLaterPages=page)
    print("Rendered",ROOT/"Report.pdf")


if __name__=="__main__":
    figures()
    render()
