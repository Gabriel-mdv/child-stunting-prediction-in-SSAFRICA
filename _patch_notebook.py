Apply llneookpch— celsoslis fins(t-qes).linelinelinelinedefpatch(idx, lines):cls[ix]["sore"] = le
 es[idx]["outputs"]=[]
    cls[idx]["xeuin_count"] = None
 CELL 9: dual threshold functions patch(,[
   ------------------------------------------------------\n",    "\n",    "\n",    "\n", # Mscore \n,"    \n","    \n","    \n","    \n","    \n","    \n","    \n","    \n",    "\n",    "\n", # M\n,"    \n","    \n","    \n","    \n","    \n","    \n","    \n","    \n",    "\n",    "\n","    \n","    eturndict(\n",
   "        thresholdt,\n",
   "        =recall,\n","     =precision,\n","        ,\n","        ,\n"," )\"
  "\n",    "''\n", #;,\n",\n",    "\n",    "\n",    "\n",    "'--'''\n,    "'\ set)')\n",
    "    print(f\"  {'Strategy':<30} {'Threshold':>10} {'Recall':>8} {'Precision':>10} {'F1':>7} {'F2':>7}\")\n",
    "    print('  ' + '-' * 72)\n",
    "    print(f\"  {'Max-F2':<30} {m_f2['threshold']:>10.3f} {m_f2['recall']:>8.3f} {m_f2['precision']:>10.3f} {m_f2['f1']:>7.3f} {m_f2['f2']:>7.3f}\")\n",
    "    lbl2 = f'Max-F1 | Recall>={RECALL_TARGET}'\n",
    "    print(f\"  {lbl2:<30} {m_f1r['threshold']:>10.3f} {m_f1r['recall']:>8.3f} {m_f1r['precision']:>10.3f} {m_f1r['f1']:>7.3f} {m_f1r['f2']:>7.3f}\")\n",
    "    if m_f2['recall'] >= m_f1r['recall']:\n",
    "        chosen, chosen_name = m_f2, 'Max-F2'\n",
    "    else:\n",
    "        chosen, chosen_name = m_f1r, f'Max-F1|Recall>={RECALL_TARGET}'\n",
    "    print(f\"  => Selected: {chosen_name}  (threshold={chosen['threshold']:.3f}  recall={chosen['recall']:.3f})\")\n",
    "    return chosen['threshold'], chosen_name, {'f2': m_f2, 'f1r': m_f1r}\n",
    "\n",
    "print('\\u2713 Threshold functions defined (F2-optimal + Max-F1|Recall>=0.75 +elcor)'\n,
]