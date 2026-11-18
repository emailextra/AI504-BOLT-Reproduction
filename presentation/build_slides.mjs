
const REPO = 'https://github.com/emailextra/AI504-BOLT-Reproduction';
const PAPER = 'https://aclanthology.org/2026.findings-acl.667/';
const FONT = 'Arial';
const C = { ink: '#172E47', muted: '#526477', blue: '#245C87', orange: '#B65E2B', pale: '#F1F5F8' };

function text(slide, value, x, y, w, h, size=28, color=C.ink, bold=false, font=FONT) {
  const shape = slide.shapes.add({geometry:'textbox',name:value.slice(0,45),
    position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  shape.text = value;
  shape.text.style = {fontFamily:font,fontSize:size,color,bold};
  shape.text.insets = 0;
  return shape;
}

function base(p, title, pointer, number) {
  const slide=p.slides.add({width:1280,height:720});
  slide.background.fill='#FFFFFF';
  text(slide,title,64,48,1152,100,44,C.ink,true);
  text(slide,pointer,64,663,1090,32,16,C.muted);
  text(slide,String(number),1170,663,46,32,16,C.muted);
  return slide;
}

function table(slide, headers, rows, y=190, height=300, widths=null) {
  const matrix=[headers,...rows];
  const t=slide.tables.add({rows:matrix.length,columns:headers.length,left:64,top:y,width:1152,height,
    columnWidths:widths||Array(headers.length).fill(1152/headers.length),values:matrix});
  for(let r=0;r<matrix.length;r++) for(let c=0;c<headers.length;c++) {
    const cell=t.getCell(r,c);
    cell.fill=r===0?C.ink:(r%2===1?C.pale:'#FFFFFF');
    cell.text.style={fontFamily:FONT,fontSize:22,color:r===0?'#FFFFFF':C.ink,bold:r===0};
  }
  return t;
}

function note(slide,seconds,content,sources=[]) {
  slide.speakerNotes.text=`Target duration ${seconds} seconds.\n\n${content}\n\nEvidence and sources\n${sources.join('\n')}`;
}

export async function build(p, details) {
  p.theme.defaultFont=FONT;
  if(p.slides.items.length) p.slides.delete(p.slides.items.map(s=>s.id));
  let s=base(p,'Reproducing BOLT on BANKING77','AI504 Assignment 2 · Group '+Number(details.group_number),1);
  text(s,'Open-set intent classification\nand a supervision-mask audit',64,175,1100,145,50,C.ink,true);
  text(s,'BOLT: Benchmarking Open-World Learning\nfor Text Classification',64,340,1120,82,29,C.blue,true);
  text(s,'Chuan Qin · Xi Chen · Jinpeng Li · Hengshu Zhu\nFindings of ACL 2026',64,435,1100,72,25,C.muted);
  const identities=details.members.map(m=>m.name+' ('+m.id+')');
  text(s,identities.slice(0,2).join(' · ')+'\n'+identities.slice(2).join(' · '),64,550,1152,86,22,C.ink);
  note(s,60,'Introduce Group 4 and the original paper, then explain the bounded reproduction. Explain that requests can be meaningful banking questions while their intent category is unknown to the classifier.',[PAPER]);

  s=base(p,'The selected DeepUNK claim','Project | Report §1 and Table 1; code/published.csv',2);
  text(s,'At 10% labeled data, increasing known classes changes both recognition and rejection.',64,150,1120,78,29);
  table(s,['Known intents','Metric','BOLT result'],[
    ['25%','Known-class F1','37.49'],['25%','Unknown-class F1','4.16'],
    ['50%','Known-class F1','48.99'],['50%','Unknown-class F1','31.21']],255,300,[240,570,342]);
  text(s,'We test these four values on the five official class partitions.',64,580,1120,54,26,C.blue);
  note(s,75,'State the claim in one sentence with dataset, label ratio, class ratios, and headline numbers. Explain that our target is a limited BANKING77 result. Both published DeepUNK metrics increase for this change. Do not generalize this trend to all BOLT methods.',[PAPER+' Tables S4 and S7','Report §1']);

  s=base(p,'Related work and the implementation gap','Project | Report §2; report/references.bib; Papers/',3);
  text(s,'Novelty in text classification',64,165,1120,45,30,C.blue,true);
  text(s,'DOC (2017), Larson et al. (2019), BANKING77 (2020)\nDefine rejection and realistic fine-grained intent evaluation.',64,220,1120,88,26);
  text(s,'Learning representations for rejection',64,330,1120,45,30,C.blue,true);
  text(s,'DeepUNK (2019), KNNCon (2022), GloVe (2014), BERT (2019)\nConnect learned features with classification and outlier detection.',64,385,1120,88,26);
  text(s,'BOLT (2026) standardizes the settings. Our audit checks which labeled rows the released code actually admits.',64,515,1120,95,29,C.ink,true);
  note(s,75,'Synthesize by theme. Explain why class novelty, representation learning, and execution provenance are different aspects of reproducibility. The gap is whether the named low-supervision condition matches selected rows in the inspected checkout.',[
    PAPER,'https://aclanthology.org/D17-1314/','https://aclanthology.org/D19-1131/','https://aclanthology.org/2020.nlp4convai-1.5/',
    'https://aclanthology.org/P19-1548/','https://aclanthology.org/2022.acl-long.352/','https://aclanthology.org/D14-1162/','https://aclanthology.org/N19-1423/']);

  s=base(p,'Data and reproduction protocol','Project | Report §3; code/train.py; results/*/run.json',4);
  table(s,['Design choice','Executed protocol'],[
    ['Dataset','BANKING77 with 3,080 held-out requests'],
    ['DeepUNK','25% and 50% known classes; five partitions each'],
    ['KNNCon sensitivity','25% known classes; partition zero; five epochs'],
    ['Supervision and seed','Nominal labeled ratio 0.10; seed 2025'],
    ['Hardware','Colab Tesla T4; isolated Python 3.11 environments']],175,365,[320,832]);
  text(s,'Fold standard deviation measures variation in class composition.',64,568,1120,55,26,C.blue);
  note(s,80,'Explain the known inventory and pooled unknown class. Distinguish ten main DeepUNK trials from two KNNCon sensitivity runs. DeepUNK uses GloVe and BiLSTM in the inspected implementation. KNNCon fine-tunes BERT. Describe fixed source revision and why fold variation is different from random-seed variation.',[PAPER,'Report §§3.1–3.4','results/environment.txt']);

  s=base(p,'The released label filter changes supervision','Project | Report §§3.1–3.2 and 4.2; results/tables/mask_audit.csv',5);
  text(s,'Released filter: .astype(bool)\nCorrected filter: .eq("True")',64,165,1120,100,30,C.blue,false,'Courier New');
  table(s,['Partition zero, 25% known','Released','Corrected'],[
    ['Training examples admitted','2,303','229'],['Development examples admitted','258','28']],300,190,[640,256,256]);
  text(s,'The paired experiment changes data access, validation selection, and the number of optimizer updates.',64,525,1120,90,28);
  note(s,70,'Explain that nonempty False strings are true when coerced to Boolean. Point to the stored masks and the anchored patch. Explain the confounding mechanisms before showing model results. A corrected five-epoch result does not isolate the value of extra labels or demonstrate convergence.',['code/author_runner.py stage_method','results/tables/mask_audit.csv','Report §5.1']);

  s=base(p,'Repository and code provenance','Code | README §6; code/; tests/; results/',6);
  text(s,'code/\n  train.py\n  author_runner.py\n  analyze_results.py\nresults/\n  full/\n  short_5epoch/\n  tables/\ntests/',64,165,550,425,26,C.ink,false,'Courier New');
  text(s,'Reused from BOLT\nTraining, tokenization, validation, and scoring implementations',690,170,500,130,29,C.blue,true);
  text(s,'Added for this study\nMask correction, prediction export, independent metric checks, tables, and error analysis',690,355,500,190,28);
  note(s,30,'Orient the marker in the public repository. Open the README. Explain that author training sources download at a pinned revision and that the project code stages instrumentation and a separate correction. Use one example function if asked.',[REPO,'README §6']);

  s=base(p,'Reproducing the reported result','Code | README §§4–5; scripts/reproduce_tables.sh; scripts/reduced_run.sh',7);
  text(s,'Main table command',64,165,1120,50,31,C.blue,true);
  text(s,'python code/analyze_results.py',64,230,1120,60,32,C.ink,false,'Courier New');
  text(s,'Run live and open results/tables/deepunk_comparison.csv.\nThe command rebuilds scores from the saved predictions.',64,315,1120,100,28);
  text(s,'Reduced GPU demonstration',64,455,1120,45,30,C.blue,true);
  text(s,'python code/train.py quick --output results/demo',64,515,1120,52,25,C.ink,false,'Courier New');
  text(s,'Measured model runtime 176.59 seconds on Tesla T4',64,585,1120,40,24,C.muted);
  note(s,180,'Use these three minutes for a real live demonstration or an authentic screen recording. Run the main analysis command and inspect the generated comparison table and prediction audit. If GPU setup is ready, a reduced model demonstration can take about three minutes, so rehearse and manage the strict block timing. Saved logs provide the complete training evidence. Never present saved console output as a newly executed training run.',[REPO,'results/quick/knncon_0.25_0/run.json','README §§4–5']);

  s=base(p,'Tests and traceable experiment evidence','Code | tests/test_reproduction.py; results/manifest.csv',8);
  text(s,'python -m unittest discover -s tests -v',64,175,1120,65,31,C.ink,false,'Courier New');
  text(s,'Checks cover the mask regression, recorded data counts, metrics, saved predictions, and portable paths.',64,285,1120,110,30);
  text(s,'16 metric rows verified\n3,080 test predictions per row',64,445,1120,115,40,C.blue,true);
  note(s,30,'Show the actual test output. Explain that checks establish internal consistency and input provenance. They do not establish convergence or independent replication of all training results. Mention the GitHub verification run and immutable evidence checksums.',['tests/test_reproduction.py','results/tables/prediction_audit.csv','results/manifest.csv']);

  s=base(p,'DeepUNK reproduction results','Report | Report §4.1, Table 1 and Figure 1; results/tables/deepunk_comparison.csv',9);
  table(s,['Known','Metric','Published','Reproduced ± fold SD'],[
    ['25%','Known F1','37.49','36.42 ± 1.66'],['25%','Unknown F1','4.16','1.19 ± 1.21'],
    ['50%','Known F1','48.99','47.77 ± 2.59'],['50%','Unknown F1','31.21','32.99 ± 12.49']],185,305,[160,275,240,477]);
  text(s,'Three of four comparisons are within two points.\nThe 25% unknown-class score differs by 2.97 points.',64,530,1120,96,29,C.blue,true);
  note(s,90,'Discuss both agreement and the gap. Unknown-class F1 has substantial fold variation at 50% known classes. Our metric audit rules out scale conversion as the cause of the discrepancy, but it cannot identify the authors\' exact initialization and checkpoint history. Conclude partial support for the selected claim.',[PAPER,'Report §§4.1 and 5.1']);

  s=base(p,'KNNCon sensitivity to the label mask','Report | Report §4.2, Table 2 and Figure 2; results/tables/knncon_paired.csv',10);
  const chart=s.charts.add('bar',{position:{left:64,top:170,width:1152,height:345},
    categories:['Known-class F1','Unknown-class F1'],
    series:[{name:'Released mask',values:[69.80,81.59],fill:C.blue},{name:'Corrected mask',values:[11.60,14.34],fill:C.orange}],
    hasLegend:true,legend:{position:'bottom',textStyle:{fontFamily:FONT,fontSize:22}},
    barOptions:{direction:'column',grouping:'clustered'},
    dataLabels:{showValue:true,position:'outEnd',textStyle:{fontFamily:FONT,fontSize:22}},
    xAxis:{textStyle:{fontFamily:FONT,fontSize:22}},
    yAxis:{minimumScale:0,maximumScale:100,textStyle:{fontFamily:FONT,fontSize:20},majorGridlines:{fill:'#DEE5ED',width:1}}});
  text(s,'Cosine LOF · one partition · one seed · five epochs\nThe correction also reduces update count and validation access.',64,555,1120,76,25,C.muted);
  note(s,60,'Explain the large paired change and its limits. Both metrics decrease at the tested budget. The training and validation sets become smaller together. A longer corrected run or equal-update comparison could behave differently. Euclidean results are retained in the report, including zero unknown F1 after correction.',['results/tables/knncon_paired.csv','Report §§4.2 and 5.1']);

  s=base(p,'Concrete failures in banking requests','Report | Report §5.2; results/tables/failure_examples.csv',11);
  const examples=[
    ['Unknown request accepted','Can I exchange currencies with this app?','exchange_via_app → fiat_currency_support'],
    ['Familiar intent confused','What countries can I travel to and have access to their currencies?','fiat_currency_support → beneficiary_not_allowed'],
    ['Familiar request rejected','Where are my funds being sourced from?','verify_source_of_funds → unknown']];
  for(let i=0;i<examples.length;i++) {
    const y=155+i*158;
    text(s,examples[i][0],64,y,1120,39,29,C.blue,true);
    text(s,examples[i][1],64,y+44,1120,62,27);
    text(s,examples[i][2],64,y+108,1120,35,23,C.muted);
  }
  note(s,70,'These are actual DeepUNK test failures from partition zero at 25% known classes. Explain each category and why the prediction is wrong under the inventory. Shared currency language is a plausible explanation for the first confusion, while the causal explanations remain hypotheses. Highlight different customer-service costs of acceptance and rejection.',['results/full/deepunk_0.25_0/predictions.csv','Report §5.2']);

  s=base(p,'Limits, resources and verdict','Report | Report §§4.3 and 5–6; results/tables/runtime.csv',12);
  text(s,'Evidence limits',64,165,1120,45,31,C.blue,true);
  text(s,'One dataset and one random seed.\nFive DeepUNK class partitions; one KNNCon partition.\nThe five-epoch mask study changes optimization exposure.',64,225,1120,150,29);
  text(s,'Measured successful model time',64,420,1120,45,31,C.blue,true);
  text(s,'19.97 minutes DeepUNK + 12.03 minutes paired KNNCon\nSetup, interrupted runs, and downloads are additional.',64,480,1120,100,28);
  note(s,50,'State the limits and resource exclusions accurately. Discuss ethical implications of misrouting a banking request and the usefulness of a service fallback. Give the verdict that the chosen DeepUNK claim is partly supported and the supervision discrepancy is established in the inspected checkout. No claim about all BOLT tasks or all implementations follows.',['Report §§4.3, 5.3–5.4 and 6']);

  s=base(p,'Partial reproduction with an evaluated code discrepancy','Report | Report §6; public repository tagged submission',13);
  text(s,'DeepUNK broadly tracks the selected published result.\nThe released KNNCon filter admits rows marked False.',64,200,1120,150,39,C.ink,true);
  text(s,REPO,64,435,1120,90,28,C.blue);
  text(s,'Questions and individual discussion',64,565,1120,50,28,C.muted);
  note(s,30,'Close with one verdict and the repository. Invite the ten-minute Q&A. Every member should be ready to explain a part they worked on and another part of the experiment. Point the marker to the report, source, or saved result for each answer.',[REPO,'Report §6']);
  const speakers=[0,0,0,1,1,1,2,2,2,3,3,3,3];
  for (let i=0;i<p.slides.items.length;i++) {
    const member=details.members[speakers[i]];
    const slide=p.slides.items[i];
    slide.speakerNotes.text='Presenter '+member.name+' ('+member.id+').\n\n'+slide.speakerNotes.text;
  }
  return {slides:p.slides.items.length,totalSeconds:900};
}
