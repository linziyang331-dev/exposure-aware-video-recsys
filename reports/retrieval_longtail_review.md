# 长尾召回模块独立审查与修复

日期：2026-10-29。按requesting-code-review流程由独立只读审查者`longtail_review`检查新增模块、配置、测试与设计。审查对象为训练前/训练中的代码版本，审查者未独立执行正式GPU实验；最终全量训练、重放、修复验证由主助手完成，不称第二次独立GPU审查。

## 审查结果

无Critical；2项Important：

1. selection.json只验证引用文件哈希，没有将自身副本绑定已冻结的trials.json；修改selected_trial仍可能通过入口。修复后要求完整选择payload等于被哈希的trials.json，且每个内嵌运行与被哈希training.json一致，复制指标不能各自为准。`test_selection_payload_cannot_diverge_from_frozen_trials`已实际先失败后通过。
2. reproduce可能覆盖损坏的复现摘要，verify未要求完整控制/候选集合，也未重算实际重训等价。修复后已完成摘要必须先核验，不能静默替换；要求精确模型集合，重读原始/新组件、实际模型参数、每轮损失与rank指标及完整验证排名。空集合拒绝测试RED→GREEN，实际CPU独立重训证明有效记录通过、缺失/修改的复现声明拒绝且损坏文件字节不变。

14项本轮测试和166项全量测试通过（1项预先存在的符号链接权限skip）。日志：`environment_logs/longtail_review_selection_red.txt`、`longtail_review_reproduction_red.txt`、`longtail_review_fixes_with_reproduction.txt`、`longtail_all_tests_review_fixed.txt`。

## 未在本轮更改的Minor

- 单独evaluate阶段在完成评分后才读取必需diagnosis.json；应增加更早前置检查。正常all顺序及本轮分阶段实际命令先完成诊断，本轮不受影响。
- 全局报告镜像不在实验目录局部report_manifest内；本次交付另做镜像逐字节核对，自动CLI核验保证范围为局部报告。未把镜像校验称持续自动保证。
- 本机Python3.12导入report.py有命令示例反斜杠的SyntaxWarning，不影响实际输出；冻结后未改源码，只记录此提示。

## 审查范围之外与最终处理

审查者未判断实际指标增益、GPU确定性和最终完整资产。主助手随后完成8配置、三种子控制/候选、主种子独立目录重训、24个完整rank重算和12288条检查点重放；明确得到“未解决长尾Top20，不采用为正式召回”的结论，见[效果评估](retrieval_longtail_assessment.md)。

审查修复只涉及选择/复现核验与报告入口，没有改变训练/model/contracts。原审查前目录和选择记录保留；新正式目录`retrieval_longtail_main_v2`只读复用经相同训练签名核验的8组运行并由新进程重新冻结，没有增加配置或按测试另选。
