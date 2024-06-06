import pandas as pd

path = 'D:/tcga-brca/BRCA-data.csv'
df = pd.read_csv(path)
columns_mapping = {'additional_studies': '额外检查',
                   'tumor_tissue_site': '肿瘤组织位置',
                   'other_dx': '其他诊断',
                   'gender': '性别',
                   'vital_status': '生存状态',
                   'Age': '年龄',
                   'days_to_death': '去世间隔天数',
                   'days_to_last_followup': '随访时间',
                   'race_list': '种族',
                   'bcr_patient_barcode': '病人条形码',
                   'tissue_source_site': '组织部位',
                   'patient_id': '病人id',
                   'bcr_patient_uuid': 'uuid',
                   'history_of_neoadjuvant_treatment': '新辅助治疗',
                   'informed_consent_verified': '知情同意验证',
                   'icd_o_3_site': '肿瘤原发部位',
                   'icd_o_3_histology': '肿瘤组织类型',
                   'icd_10': '编码系统',
                   'year_of_initial_pathologic_diagnosis': '初次诊断年份',
                   'person_neoplasm_cancer_status': '肿瘤状态',
                   'primary_lymph_node_presentation_assessment': '初次淋巴结状态',
                   'lymph_node_examined_count': '淋巴结数量',
                   'er_detection_method_text': 'ER检测',
                   'pgr_detection_method_text': 'PR检测',
                   'breast_carcinoma_progesterone_receptor_status': '孕激素受体',
                   'anatomic_neoplasm_subdivisions': '解剖位置',
                   'her2_neu_chromosone_17_signal_ratio_value': 'HER2状态检测',
                   'axillary_lymph_node_stage_method_type': 'ALN分期检测',
                   'axillary_lymph_node_stage_other_method_descriptive_text': 'ALN分期检测补充',
                   'breast_carcinoma_surgical_procedure_name': '手术类型',
                   'breast_neoplasm_other_surgical_procedure_descriptive_text': '手术类型补充',
                   'breast_carcinoma_primary_surgical_procedure_name': '外科手术名',
                   'surgical_procedure_purpose_other_text': '额外手术信息',
                   'histological_type': '常规组织学特征',
                   'histological_type_other': '非常规组织学特征',
                   'menopause_status': '绝经状态',
                   'cytokeratin_immunohistochemistry_staining_method_micrometastasis_indicator': '细胞角蛋白免疫组化微小转移检测法',
                   'breast_carcinoma_immunohistochemistry_er_pos_finding_scale': '乳腺癌ER免疫组化阳性评分量表',
                   'immunohistochemistry_positive_cell_score': '免疫组化阳性细胞评分',
                   'her2_immunohistochemistry_level_result': 'HER2免疫组化水平检测结果',
                   'breast_cancer_surgery_margin_status': '乳腺癌手术切缘状态',
                   'margin_status': '切缘状态',
                   'initial_pathologic_diagnosis_method': '初始病理诊断方法',
                   'lab_procedure_her2_neu_in_situ_hybrid_outcome_type': '其他初始病理诊断方法',
                   'breast_carcinoma_estrogen_receptor_status': '乳腺癌雌激素受体状态',
                   'lab_proc_her2_neu_immunohistochemistry_receptor_status': 'HER2免疫组化受体状态',
                   'number_of_lymphnodes_positive_by_ihc': '阳性淋巴结数',
                   'number_of_lymphnodes_positive_by_he': 'H&E染色阳性淋巴结数',
                   'pos_finding_progesterone_receptor_other_measurement_scale_text': 'PR阳性其他测量',
                   'positive_finding_estrogen_receptor_other_measurement_scale_text': 'ER阳性其他测量描述',
                   'her2_erbb_pos_finding_cell_percent_category': 'HER2阳性细胞百分比分类',
                   'pos_finding_her2_erbb2_other_measurement_scale_text': 'HER2其他检测描述',
                   'her2_erbb_method_calculation_method_text': 'HER2检测计算方法',
                   'her2_neu_and_centromere_17_copy_number_analysis_input_total_number_count': 'HER2/CEP17拷贝数计数',
                   'her2_and_centromere_17_positive_finding_other_measurement_scale_text': 'HER2/CEP17其他测量结果',
                   'her2_erbb_pos_finding_fluorescence_in_situ_hybridization_calculation_method_text': 'HER2 FISH计算方法',
                   'tissue_prospective_collection_indicator': '组织前瞻性收集指标',
                   'fluorescence_in_situ_hybridization_diagnostic_procedure_chromosome_17_signal_result_range': 'FISH染色体17信号范围',
                   'first_nonlymph_node_metastasis_anatomic_sites': '非淋巴结首转部位',
                   'er_level_cell_percentage_category': 'ER水平细胞百分比分类',
                   'progesterone_receptor_level_cell_percent_category': 'PR水平细胞百分比分类',
                   'metastatic_breast_carcinoma_estrogen_receptor_status': '转移性乳腺癌ER状态',
                   'metastatic_breast_carcinoma_estrogen_receptor_level_cell_percent_category': '转移性乳腺癌ER表达分类',
                   'metastatic_breast_carcinoma_immunohistochemistry_er_pos_cell_score': '转移性乳腺癌IHC评分',
                   'metastatic_breast_carcinoma_progesterone_receptor_status': '转移性乳腺癌PR状态',
                   'metastatic_breast_carcinoma_lab_proc_her2_neu_immunohistochemistry_receptor_status': '转移性乳腺癌HER2 IHC状态',
                   'metastatic_breast_carcinoma_progesterone_receptor_level_cell_percent_category': '转移性乳腺癌孕激素受体水平细胞百分比类别',
                   'metastatic_breast_carcinoma_immunohistochemistry_pr_pos_cell_score': '转移性乳腺癌免疫组化孕激素受体阳性细胞评分',
                   'metastatic_breast_carcinoma_her2_erbb_pos_finding_cell_percent_category': '转移性乳腺癌HER2/ERBB阳性细胞百分比类别',
                   'metastatic_breast_carcinoma_erbb2_immunohistochemistry_level_result': '转移性乳腺癌ERBB2免疫组化水平结果',
                   'metastatic_breast_carcinoma_lab_proc_her2_neu_in_situ_hybridization_outcome_type': '转移性乳腺癌实时荧光原位杂交结果类型',
                   'metastatic_breast_carcinoma_her2_neu_chromosone_17_signal_ratio_value': '转移性乳腺癌HER2/neu染色体17信号比值',
                   'her2_neu_breast_carcinoma_copy_analysis_input_total_number': '乳腺癌HER2/neu拷贝分析输入总数',
                   'breast_carcinoma_immunohistochemistry_progesterone_receptor_pos_finding_scale': '乳腺癌免疫组化孕激素受体阳性发现评分',
                   'breast_carcinoma_immunohistochemistry_pos_cell_score': '乳腺癌免疫组化阳性细胞分数',
                   'distant_metastasis_present_ind':'远处转移存在指标',
                   'tissue_retrospective_collection_indicator':'组织样本回顾性收集指标',
                   'init_pathology_dx_method_other':'病理诊断方法',
                   'stage_event': '分期事件',
                   'postoperative_rx_tx': '术后治疗',
                   'radiation_therapy': '放射治疗',
                   'new_tumor_events': '新肿瘤事件',
                   'day_of_form_completion': '填表日期',
                   'month_of_form_completion': '填表月份',
                   'year_of_form_completion': '填表年份',
                   'follow_ups': '随访',
                   'drugs': '药物',
                   'radiations': '放疗'
                   }


# df.rename(columns=columns_mapping, inplace=True)

# 将重命名后的DataFrame保存为新的CSV文件
# df.to_csv('D:/毕设/py-django/flaskProject/data/data_final.csv',encoding='utf-8', index=False)