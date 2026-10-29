# 统一系统评估独立审查与修复

日期：2026-10-29。独立审查者在单独上下文只读检查`src/system_evaluation`、配置、测试及设计/计划；没有修改文件，没有重跑全量排名、GPU抽样或深度曝光/RL实验。项目无Git，按文件及签名审查。

## 独立检查

数值契约、正事件/用户对齐、冷目标保留、用户宏平均、固定候选集合、互斥完整漏斗与四类任务分离实现合理。独立执行10项指标/接口CPU测试全部通过；查看了初版smoke的输入清单和四分区抽样元数据。没有发现Critical。

独立审查发现三项Important，均由主代理在同一轮补充失败回归并修复：

1. 原verification只绑定输入指纹和manifest，报告信任其中未核对的重算数量/布尔字段。审查者复现了假计数被接受。现在从当前analysis、pipeline、config及组件重建完整verification记录，报告读取前逐字段核对。
2. 原verify不核对外层report、summary、figures和正式镜像。审查者在隔离小组件中复现篡改报告/新收益字段后仍PASS。现在增加独立`report_delivery.json`，绑定评估manifest、verification和全部交付文件；已有报告的verify先校验交付清单；report重新生成后刷新清单并自检。独立复现目录不覆盖正式镜像。
3. 原预算句只根据smoke判断深度重放，允许的deep_audit=false非smoke配置会误报。现在按实际deep_audit生成预算描述，并核对完整验证记录。

另落实两项报告清晰性建议：V3/V2标准测试Recall由真实记录动态生成；响应一致性明确限于检索batch512/精排batch256，其他批形只计时，不宣称跨批形浮点或Top20完全相同。

## 主代理回归和实际验收

主代理保留上述失败日志：`environment_logs/system_review_regressions_red.txt`；CSV完整性与计时scope失败日志为`system_export_scope_red.txt`。最终145项全项目测试通过（67.879秒），本轮新增18项，见`system_all_tests_final.txt`。

全量联查还发现早期召回没有保存macro@100。追溯旧实现后，补充缺失字段回归RED→GREEN：保留从原排名重算的新值，仅对照旧文件实际保存的字段，没有把缺失当0或补造历史结果，见`system_legacy_macro_red.txt`。

最终smoke端到端运行通过；正式all完成84份排名、16份曝光预测、440行分组与2048条冻结模型GPU响应重放，深度重算曝光和CQL通过。所有四类CSV逐值核对，计时只核验原始观测/次数/分位汇总与scope，没有宣称精确重放。

两个正式PNG已由主代理目视检查；报告表格与漏斗来源为当前实际重算数据。此处完整运行与修复验收来自主代理，不表示独立审查者复审了后续修复。

## 适用边界

本阶段交付统一核验、离线接口抽样和诊断，未训练新模型、未改变正式Top20。全目录MRR、池内MRR@100、真实概率预测、模拟OPE/RL不可混算。真实测试已查看，只作迭代复测；计时不含历史查询、网络或并发，不是线上SLO。新环境安装/全链路重训、线上部署、真实因果收益和后续优化均未被本阶段证明。

独立审查对这些明确排除的领域未作质量判断；主代理保留同一边界。

## 最终交付证据补充

主代理在独立目录重新执行完整all，全部质量结果与非计时响应元数据一致；analysis、四份CSV、四份probe共9个确定性文件SHA256完全相同。计时保留为新的观测，图表/SVG不要求逐字节复现，见[同机复查](system_evaluation_reproduction.json)。独立目录没有覆盖正式报告镜像。

实际再次执行已有交付的`--phase verify`通过，之后报告清单仍然有效，覆盖审查发现的验收分支。完成目录复用不重写评估与交付manifest，见[复用证据](system_evaluation_component_reuse.json)。最终[交付核验](system_evaluation_delivery_validation.json)为PASS，包含620个输入、8个报告交付输出、环境与镜像一致性；正式完整重算日志为`environment_logs/system_full.txt`，既有交付verify日志为`system_verify_existing_delivery.txt`。这些证据来自主代理执行，不表示独立审查者完成了GPU或全量重算。
