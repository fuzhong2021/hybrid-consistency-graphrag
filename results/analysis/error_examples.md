# Qualitative Beispielanalyse (Phase 6)

Stichproben aus den Logs ohne Ground-Truth-Verknüpfung. Daher hier *Beispielentscheidungen* nach interessanten Mustern, keine echten False-Positive/False-Negative-Klassifikationen — diese bräuchten einen instrumentierten Re-Lauf (siehe `evaluation/analysis/instrument_logging_patch.md`).

## `full_evaluation_with_llm.log`

### Stage-3 LLM-REJECT (möglicher False Negative)

- **REJECTED** [Stufe 3 fail] in 3706.8 ms (Datensatz: `hotpotqa`)
  - Triple: `End of Days (film) --[HAS_ANSWER]--> 1999`
- **REJECTED** [Stufe 3 fail] in 3176.8 ms (Datensatz: `hotpotqa`)
  - Triple: `Brown State Fishing Lake --[HAS_ANSWER]--> 9,984`
- **REJECTED** [Stufe 3 fail] in 3179.8 ms (Datensatz: `hotpotqa`)
  - Triple: `Masakazu Katsura --[RELATED_TO]--> I&quot;s`
- **REJECTED** [Stufe 3 fail] in 8086.9 ms (Datensatz: `hotpotqa`)
  - Triple: `Sullivan County, New Hampshire --[RELATED_TO]--> East Lempster, New Hampshire`
- **REJECTED** [Stufe 3 fail] in 2935.2 ms (Datensatz: `hotpotqa`)
  - Triple: `Handi-Snacks --[RELATED_TO]--> Mondelez International`

### Stage-3 LLM-ACCEPT nach Stage-2 Fast-Path-Bypass

- **ACCEPTED** [Stufe 3 pass] in 4628.7 ms (Datensatz: `hotpotqa`)
  - Triple: `Kansas Song --[RELATED_TO]--> University of Kansas`
- **ACCEPTED** [Stufe 3 pass] in 3906.2 ms (Datensatz: `hotpotqa`)
  - Triple: `End of Days (film) --[RELATED_TO]--> Oh My God (Guns N' Roses song)`
- **ACCEPTED** [Stufe 3 pass] in 2828.0 ms (Datensatz: `hotpotqa`)
  - Triple: `Brown State Fishing Lake --[RELATED_TO]--> Brown County, Kansas`
- **ACCEPTED** [Stufe 3 pass] in 3569.0 ms (Datensatz: `hotpotqa`)
  - Triple: `Peter Schmeichel --[HAS_ANSWER]--> World's Best Goalkeeper`
- **ACCEPTED** [Stufe 3 pass] in 3505.4 ms (Datensatz: `hotpotqa`)
  - Triple: `These Boots Are Made for Walkin' --[RELATED_TO]--> Lee Hazlewood`

### Stage-1 Fast-Reject (Schema/Cardinality)

- **REJECTED** [Stufe 1 FAIL] in 0.1 ms (Datensatz: `hotpotqa`)
  - Triple: `Eenasul Fateh --[HAS_ANSWER]--> Eenasul Fateh`
- **REJECTED** [Stufe 1 FAIL] in 0.1 ms (Datensatz: `hotpotqa`)
  - Triple: `Kansas Song --[HAS_ANSWER]--> Kansas Song`
- **REJECTED** [Stufe 1 FAIL] in 0.2 ms (Datensatz: `hotpotqa`)
  - Triple: `David Weissman --[HAS_ANSWER]--> David Weissman`
- **REJECTED** [Stufe 1 FAIL] in 0.1 ms (Datensatz: `hotpotqa`)
  - Triple: `Charles Nungesser --[RELATED_TO]--> L'Oiseau Blanc`
- **REJECTED** [Stufe 1 FAIL] in 0.1 ms (Datensatz: `hotpotqa`)
  - Triple: `Sachin Warrier --[RELATED_TO]--> Tata Consultancy Services`

### NLI Contradiction

- **REJECTED** [NLI CONTRADICTION] in 75.9 ms (Datensatz: `hotpotqa`)
  - Triple: `Shirley Temple --[RELATED_TO]--> Kiss and Tell (1945 film)`
- **REJECTED** [NLI CONTRADICTION] in 76.7 ms (Datensatz: `hotpotqa`)
  - Triple: `The Hork-Bajir Chronicles --[RELATED_TO]--> Animorphs`
- **REJECTED** [NLI CONTRADICTION] in 80.6 ms (Datensatz: `hotpotqa`)
  - Triple: `Jim Cummings --[HAS_ANSWER]--> Sonic`
- **REJECTED** [NLI CONTRADICTION] in 73.5 ms (Datensatz: `hotpotqa`)
  - Triple: `Poison (American band) --[HAS_ANSWER]--> 2000`
- **REJECTED** [NLI CONTRADICTION] in 94.5 ms (Datensatz: `hotpotqa`)
  - Triple: `Ralph Hefferline --[RELATED_TO]--> Columbia University`

### Hohe Latenz (>5000 ms)

- **REJECTED** [Stufe 3 fail] in 6779.5 ms (Datensatz: `hotpotqa`)
  - Triple: `Harry Connick Jr. --[RELATED_TO]--> When Harry Met Sally... (soundtrack)`
- **REJECTED** [Stufe 3 fail] in 6099.7 ms (Datensatz: `hotpotqa`)
  - Triple: `Maroon 5 --[RELATED_TO]--> What Lovers Do`
- **REJECTED** [Stufe 3 fail] in 5429.2 ms (Datensatz: `hotpotqa`)
  - Triple: `Robert Parker Parrott --[RELATED_TO]--> Cold Spring, New York`
- **ACCEPTED** [Stufe 3 pass] in 8177.4 ms (Datensatz: `hotpotqa`)
  - Triple: `Thomas H. Ince --[RELATED_TO]--> Joseph McGrath (film director)`
- **REJECTED** [Stufe 3 fail] in 7069.8 ms (Datensatz: `hotpotqa`)
  - Triple: `Raymond Ochoa --[RELATED_TO]--> The Good Dinosaur`

## `full_evaluation_no_llm.log`

### Stage-3 LLM-REJECT (möglicher False Negative)

_keine Beispiele in diesem Log_

### Stage-3 LLM-ACCEPT nach Stage-2 Fast-Path-Bypass

_keine Beispiele in diesem Log_

### Stage-1 Fast-Reject (Schema/Cardinality)

- **REJECTED** [Stufe 1 FAIL] in 0.1 ms (Datensatz: `hotpotqa`)
  - Triple: `Eenasul Fateh --[HAS_ANSWER]--> Eenasul Fateh`
- **REJECTED** [Stufe 1 FAIL] in 0.1 ms (Datensatz: `hotpotqa`)
  - Triple: `Kansas Song --[RELATED_TO]--> University of Kansas`
- **REJECTED** [Stufe 1 FAIL] in 0.1 ms (Datensatz: `hotpotqa`)
  - Triple: `David Weissman --[RELATED_TO]--> The Family Man`
- **REJECTED** [Stufe 1 FAIL] in 0.1 ms (Datensatz: `hotpotqa`)
  - Triple: `Poison (American band) --[HAS_ANSWER]--> 2000`
- **REJECTED** [Stufe 1 FAIL] in 0.1 ms (Datensatz: `hotpotqa`)
  - Triple: `Sachin Warrier --[RELATED_TO]--> Tata Consultancy Services`

### NLI Contradiction

- **REJECTED** [NLI CONTRADICTION] in 74.2 ms (Datensatz: `hotpotqa`)
  - Triple: `Shirley Temple --[RELATED_TO]--> Kiss and Tell (1945 film)`
- **REJECTED** [NLI CONTRADICTION] in 70.1 ms (Datensatz: `hotpotqa`)
  - Triple: `The Hork-Bajir Chronicles --[RELATED_TO]--> Animorphs`
- **REJECTED** [NLI CONTRADICTION] in 74.8 ms (Datensatz: `hotpotqa`)
  - Triple: `Henry Roth --[HAS_ANSWER]--> Robert Erskine Childers DSC`
- **REJECTED** [NLI CONTRADICTION] in 72.0 ms (Datensatz: `hotpotqa`)
  - Triple: `Tunnels &amp; Trolls --[RELATED_TO]--> Arena of Khazan`
- **REJECTED** [NLI CONTRADICTION] in 70.2 ms (Datensatz: `hotpotqa`)
  - Triple: `Ralph Hefferline --[RELATED_TO]--> Columbia University`

### Hohe Latenz (>5000 ms)

_keine Beispiele in diesem Log_

## `musique_evaluation.log`

### Stage-3 LLM-REJECT (möglicher False Negative)

- **REJECTED** [Stufe 3 fail] in 3452.2 ms (Datensatz: `musique`)
  - Triple: `Ashkenazi Jews --[RELATED_TO]--> Ashkenazi Jews`
- **REJECTED** [Stufe 3 fail] in 2642.1 ms (Datensatz: `musique`)
  - Triple: `California Gold Rush --[RELATED_TO]--> California Gold Rush`
- **REJECTED** [Stufe 3 fail] in 3476.4 ms (Datensatz: `musique`)
  - Triple: `Stade de ASC HLM --[HAS_ANSWER]--> Senegal`
- **REJECTED** [Stufe 3 fail] in 2984.5 ms (Datensatz: `musique`)
  - Triple: `Rialto Bridge --[RELATED_TO]--> Orlando furioso (Vivaldi, 1714)`
- **REJECTED** [Stufe 3 fail] in 3060.5 ms (Datensatz: `musique`)
  - Triple: `George Hollis (footballer) --[RELATED_TO]--> 1894–95 FA Cup`

### Stage-3 LLM-ACCEPT nach Stage-2 Fast-Path-Bypass

- **ACCEPTED** [Stufe 3 pass] in 3281.8 ms (Datensatz: `musique`)
  - Triple: `Return to Nim's Island --[HAS_ANSWER]--> Steve Irwin`
- **ACCEPTED** [Stufe 3 pass] in 2994.2 ms (Datensatz: `musique`)
  - Triple: `Second City derby --[RELATED_TO]--> Matthew Webb (footballer)`
- **ACCEPTED** [Stufe 3 pass] in 4609.5 ms (Datensatz: `musique`)
  - Triple: `No Ordinary Girl --[RELATED_TO]--> Permission to Fly`
- **ACCEPTED** [Stufe 3 pass] in 4032.9 ms (Datensatz: `musique`)
  - Triple: `Doctor De Soto --[RELATED_TO]--> Shrek!`
- **ACCEPTED** [Stufe 3 pass] in 3630.2 ms (Datensatz: `musique`)
  - Triple: `I Can Only Imagine (film) --[HAS_ANSWER]--> Meg Ryan`

### Stage-1 Fast-Reject (Schema/Cardinality)

- **REJECTED** [Stufe 1 FAIL] in 0.1 ms (Datensatz: `musique`)
  - Triple: `Ashkenazi Jews --[RELATED_TO]--> Ashkenazi Jews`
- **REJECTED** [Stufe 1 FAIL] in 0.1 ms (Datensatz: `musique`)
  - Triple: `Ashkenazi Jews --[RELATED_TO]--> Ashkenazi Jews`
- **REJECTED** [Stufe 1 FAIL] in 0.0 ms (Datensatz: `musique`)
  - Triple: `Ashkenazi Jews --[RELATED_TO]--> Ashkenazi Jews`
- **REJECTED** [Stufe 1 FAIL] in 0.1 ms (Datensatz: `musique`)
  - Triple: `Jews --[RELATED_TO]--> Jews`
- **REJECTED** [Stufe 1 FAIL] in 0.0 ms (Datensatz: `musique`)
  - Triple: `Jews --[RELATED_TO]--> Jews`

### NLI Contradiction

- **REJECTED** [NLI CONTRADICTION] in 69.3 ms (Datensatz: `musique`)
  - Triple: `List of awards and nominations received by Matt Damon --[RELATED_TO]--> Dazed and Confused (film)`
- **REJECTED** [NLI CONTRADICTION] in 67.2 ms (Datensatz: `musique`)
  - Triple: `Tarot Classics --[RELATED_TO]--> Pythons (album)`
- **REJECTED** [NLI CONTRADICTION] in 65.2 ms (Datensatz: `musique`)
  - Triple: `Germans --[RELATED_TO]--> Middle Ages`
- **REJECTED** [NLI CONTRADICTION] in 48.9 ms (Datensatz: `musique`)
  - Triple: `Middle Ages --[RELATED_TO]--> Sylvester`
- **REJECTED** [NLI CONTRADICTION] in 68.0 ms (Datensatz: `musique`)
  - Triple: `Israel --[RELATED_TO]--> Saudi Arabia`

### Hohe Latenz (>5000 ms)

- **REJECTED** [Stufe 3 fail] in 6036.6 ms (Datensatz: `musique`)
  - Triple: `Middle Ages --[RELATED_TO]--> Holy Roman Empire`
- **REJECTED** [Stufe 3 fail] in 7631.5 ms (Datensatz: `musique`)
  - Triple: `Marmaduke Pickthall --[HAS_ANSWER]--> 48.8 percent`
- **REJECTED** [Stufe 3 fail] in 6699.7 ms (Datensatz: `musique`)
  - Triple: `Desert Diamond West Valley Phoenix Grand Prix --[RELATED_TO]--> Tucson, Arizona`
- **ACCEPTED** [Stufe 3 pass] in 6487.7 ms (Datensatz: `musique`)
  - Triple: `Chiang Hsiao-wu --[HAS_ANSWER]--> Chiang Hsiao-wu`
- **REJECTED** [Stufe 3 fail] in 7194.6 ms (Datensatz: `musique`)
  - Triple: `List of municipalities in Georgia (U.S. state) --[RELATED_TO]--> Capital punishment in the United States`

