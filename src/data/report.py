"""Render the data-stage report from verified machine-readable artifacts."""
import json
from datetime import datetime, timezone, timedelta

from src.data.common import ROOT


def main():
    def read(name):return json.loads((ROOT/'reports'/name).read_text(encoding='utf-8'))
    audit,prepared,verified=read('data_audit.json'),read('data_preparation.json'),read('data_validation.json')
    if prepared['status']!='PREPARED' or verified['status']!='PASS' or verified['rows_checked']!=prepared['rows_written']:
        raise ValueError('A successful full preparation and validation are required before reporting')
    lines=['# KuaiRand-Pure 数据审计与准备报告','',
           '生成时间：'+datetime.now(timezone(timedelta(hours=8))).isoformat()+'（Asia/Shanghai）。','',
           '本报告由实际文件、SQLite 审计结果和已通过的验证产物生成。没有模型训练或推荐收益结论。','',
           '## 数据来源、完整性与许可','',
           '- 官方说明：[KuaiRand](https://kuairand.com/)。下载记录：[Zenodo 10439422](https://zenodo.org/records/10439422)。',
           f'- 压缩包 {audit["download"]["archive"]["bytes"]:,} 字节；官方 MD5 `0820331067a3784d9691136f772b35a7` 已实际匹配。',
           f'- 解压文件共 {audit["download"]["extracted_bytes"]:,} 字节，各文件 SHA-256 已保存并在审计前复核。',
           '- Zenodo 元数据许可为 CC BY 4.0；压缩包内 LICENSE 为 CC BY-SA 4.0，标注存在差异，未宣称已消除该差异。原始数据和处理后逐条数据均不提交 GitHub，只保留官方获取说明。',
           '- 下载、解压、中间数据库、缓存与 Parquet 均位于 D 盘项目；未安装新的依赖。','',
           '## 原始日志与清理','',
           '| 文件 | 原始记录数 | 用户数 | 视频数 | 缺失项总数 |',
           '| --- | ---: | ---: | ---: | ---: |']
    for file in audit['files']:
        lines.append(f'| {file["file"]} | {file["rows"]:,} | {file["users"]:,} | {file["videos"]:,} | {sum(file["missing"].values()):,} |')
    lines += ['', '| 文件 | 负视频时长 | 负观看时长 | 零视频时长 | 观看超过视频时长 |',
              '| --- | ---: | ---: | ---: | ---: |']
    for file in audit['files']:
        lines.append(f'| {file["file"]} | {file["negative_duration_rows"]:,} | {file["negative_play_time_rows"]:,} | {file["zero_duration_rows"]:,} | {file["play_longer_than_video_rows"]:,} |')
    lines += ['', '观看超过视频时长可能包含重复播放，不直接视为坏样本，也不据此改写发布标签。']
    lines += ['',f'总原始记录 {audit["raw_rows"]:,}；按来源及全部 19 个原始字段进行精确去重，移除 {audit["exact_duplicates_removed"]:,} 条，保留 {audit["unique_rows"]:,} 条，涉及 {audit["total_users"]:,} 名用户、{audit["total_videos"]:,} 个视频。',
              '去重使用 SQLite 全字段 GROUP BY，不使用近似哈希计数。不同时间的重复观看、不同反馈的同键记录均保留；它们可能是不同曝光，不能按用户—视频对全部删除。',
              f'去重后，相同来源/用户/视频/毫秒/场景的额外记录仍有 {audit["same_identity_extra_rows"]:,} 条；这些记录的其他原始字段不同，历史构造将同毫秒组统一延后更新。',
              f'核心字段异常 {audit["invalid_core_rows"]:,} 条，文件来源与 is_rand 不一致 {audit["source_flag_mismatches"]:,} 条。未因此静默删除记录。','',
              '## 事件字段与预测目标','',
              '| 实际字段 | 用途与边界 |','| --- | --- |']
    descriptions={
        'user_id':'原始用户编号；0 是合法 ID，不是原始缺失值。',
        'video_id':'原始视频编号；编码映射只由标准训练期曝光拟合。',
        'date':'原日志日期，用于审计，不作为切分或模型输入。',
        'hourmin':'原日志时间桶，用于审计，不作为精确时序依据。',
        'time_ms':'毫秒事件时间戳；保留 int64，作为唯一时序和切分基准。',
        'is_click':'主目标：两栏界面为点击，单栏界面为有效播放，不能统称纯点击率。',
        'long_view':'长观看辅助标签；保留为标签字段，不作为当前样本输入。',
        'is_rand':'随机干预标记，用于划分来源，不能当成曝光概率。',
        'tab':'场景编号；用于场景分组与允许的上下文输入，不等于完整行为策略。',
        'play_time_ms':'观看反馈，仅审计；不作为当前目标的可用预测特征。',
        'duration_ms':'日志中的视频时长；记录异常，但当前基础输入不使用。',
        'is_like':'点赞反馈，仅审计。','is_follow':'关注反馈，仅审计。',
        'is_comment':'评论反馈，仅审计。','is_forward':'转发反馈，仅审计。',
        'is_hate':'负向反馈；与未点击、未交互严格区分。',
        'profile_stay_time':'作者页停留反馈，仅审计。',
        'comment_stay_time':'评论区停留反馈，仅审计。',
        'is_profile_enter':'进入作者页反馈，仅审计。'}
    for field in audit['columns']:lines.append(f'| `{field}` | {descriptions[field]} |')
    lines += ['', '点击/有效播放标签直接使用发布值，不根据观看时长重新制造标签。已曝光且 label=0 是观测到的未点击/未有效播放，不等价于明确不喜欢；召回阶段额外采样的未交互负样本必须单独标记。',
              '当前模型输入白名单只有 user_index、item_index、tab、history_item_indices；当前样本的点击、观看时长、点赞等反馈不进入特征。','',
              '## 来源与反馈分布','',
              '| 来源（去重后） | 记录数 | 正反馈数 | 主标签正例率 | 长观看率 |',
              '| --- | ---: | ---: | ---: | ---: |']
    for source in audit['source_summary']:
        lines.append(f'| {source["source"]} | {source["rows"]:,} | {source["positives"]:,} | {source["positive_rate"]:.4%} | {source["long_view_rate"]:.4%} |')
    lines += ['', '以下为按场景分组的观测分布，来源差异包含场景组成、样本选择和曝光机制差异，不能据此直接声称因果去偏收益。','',
              '| 来源 | tab | 记录数 | 主标签正例率 |','| --- | ---: | ---: | ---: |']
    for row in audit['source_tab_summary']:lines.append(f'| {row["source"]} | {row["tab"]} | {row["rows"]:,} | {row["positive_rate"]:.4%} |')
    lines += ['', '其他原始反馈的逐字段分布保存在 data_audit.json 的 files[].feedback；不将其他反馈与主标签混为一个目标。','',
              '## 时间异常与固定划分','',
              f'有 {audit["date_timestamp_mismatches"]:,} 条原始记录的 date 与北京时间 time_ms 日期不同，全部表现为 date 是次日，时间集中在 21–23 点。具体时钟/分桶机制未确认，不修改原始时间，也不把这些记录当作无效样本删除。',
              '按官方 time_ms 毫秒时间戳定义排序；date/hourmin 仅作审计。这是预先固定的数据协议，不声称已验证真实客户端反馈的可用时刻。标准数据最早实际时间为 2022-04-09，而文件名含 4_08，不以文件名代替实际范围。','',
              '| 分区 | 北京时间范围（左闭右开） |',
              '| --- | --- |',
              '| train | 2022-04-08 00:00 至 2022-04-22 00:00 |',
              '| validation | 2022-04-22 00:00 至 2022-04-29 00:00 |',
              '| test | 2022-04-29 00:00 至 2022-05-09 00:00 |','',
              '边界来自官方采集阶段及固定日历，不通过标签最优化选择。仅标准来源进入训练；随机来源只保留为独立来源诊断分区。测试标签不参与模型拟合或参数选择。','',
              '| 来源/分区 | 记录数 | 正反馈 | 冷用户记录 | 冷视频记录 | 空历史记录 |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for key,stat in prepared['partitions'].items():
        lines.append(f'| {key} | {stat["rows"]:,} | {stat["positives"]:,} | {stat["cold_user_rows"]:,} | {stat["cold_item_rows"]:,} | {stat["empty_history_rows"]:,} |')
    lines += ['',f'训练词表含 {prepared["train_users"]:,} 名用户、{prepared["train_items"]:,} 个视频，索引从 1 开始；0 专用于未知/填充。流行度只计标准训练期正反馈；训练曝光中没有正反馈的视频也保留。冷启动记录没有被丢弃，后续指标必须同时报告总体与适用的暖启动子集及覆盖率。','',
              '## 历史序列与信息边界','',
              '- 最长 50，落盘为变长列表，模型读取时再截断/padding；不一次生成全部样本的 Python padded 列表。',
              '- 仅严格早于当前 time_ms 的标准来源正反馈可以更新历史，且视频必须在训练词表内。随机反馈永不更新历史。',
              '- 同用户同毫秒事件全部先预测再更新历史，排除目标自身及同时间戳反馈；同时间戳在后续历史中的截断次序按稳定 event_id 确定。',
              '- 使用固定的滚动评测协议：较早验证/测试期的标准反馈可成为更晚事件的上下文；模型参数与词表不因此更新。这不等于冻结在训练结束时的静态历史。不同模型后续应采用相同的上下文协议比较。',
              '- 空历史、冷用户、冷视频仍保留；序列模型适用样本计数见 data_preparation.json 的 sequence_eligible_rows。',
              '- 目前只能证明发布事件时间的先后，不具备反馈完成/到达时间字段，不能据此证明线上反馈即时可用。','',
              '## 侧信息、缺失值与泄漏风险','',
              '| 实际特征表 | 行数 | 列数（含 ID） | 重复主键 | 日志实体缺少元数据 | 当前启用 |',
              '| --- | ---: | ---: | ---: | ---: | --- |']
    for table in audit['feature_tables']:
        lines.append(f'| {table["file"]} | {table["rows"]:,} | {len(table["columns"])} | {table["duplicate_key_rows"]} | {table["log_entities_missing_metadata"]} | 否 |')
    lines += ['', '完整字段、实际 dtype 和缺失数在 data_audit.json 的 feature_tables。非零缺失如下：','',
              '| 表 | 字段 | 缺失数 |','| --- | --- | ---: |']
    for table in audit['feature_tables']:
        for field,count in table['missing'].items():
            if count:lines.append(f'| {table["file"]} | {field} | {count:,} |')
    lines += ['', '视频统计表是跨整月与场景聚合，可能包含验证/测试期统计，默认禁止用作本项目输入或作为“曝光概率”。用户活跃度、粉丝数等快照采集时点未确立，同样不默认启用。基础视频属性可供后续单独审计后加入；本阶段不做全时期均值插补，也不把这些快照提前拼入训练数据。','',
              '## 用户活跃度与视频流行度','',
              '| 分位 | 每用户交互数 | 每视频曝光数 |','| --- | ---: | ---: |']
    for quantile,value in audit['user_interaction_quantiles'].items():
        lines.append(f'| {quantile} | {value:.2f} | {audit["video_exposure_quantiles"][quantile]:.2f} |')
    lines += ['',f'按全部观测日志统计，曝光最高的前 10% 视频占 {audit["top_10_percent_video_exposure_share"]:.4%} 的曝光。这是描述性审计；后续评测分组只能使用训练期流行度，不能复用全时期排序。','',
              '## IPS/SNIPS/DR 与 CQL 适用性','',
              '- 实际日志没有可用的逐事件 action propensity 字段；is_rand、tab、视频聚合 show_cnt 都不能替代它。全量干预比例及 Pure 文件中随机来源的占比都不等于目标动作概率。',
              '- 候选池存在不等于各时点、各用户的可行动作支持与采样概率均可重建，不能自行写入 1/7583 等概率。当前没有生成 propensity 列，也没有报告真实日志 IPS/SNIPS/DR 价值。',
              '- 官方明确 Pure 移除了池外视频交互，历史是不完整子序列。当前数据不足以证明完整状态转移或真实策略收益，CQL 后续应先验证独立原型；若需要更严格的真实序列条件，再评估是否升级数据版本。','',
              '## 实际验证与资源','',
              f'- 全量检查 {verified["rows_checked"]:,} 条：文件哈希、事件 ID 唯一与完整、原始目标字段逐项一致、来源/日历边界、训练词表与流行度、无空值、全部历史的长度/顺序/严格过去时间。',
              f'- 对五个分区合计 {verified["sampled_histories_checked"]} 条固定种子样本，用独立 SQL 从原始去重事件重建最近 K 个合格正反馈并逐项比较。此项是抽样，不是全量 SQL 历史重建。',
              '- 回归测试覆盖未来/同毫秒反馈、随机反馈隔离、冷启动、重复曝光、边界时间、缓存代码变更、原始目录绑定；人为构造哈希一致的未来历史与 NULL 标签，验证器均应拒绝。',
              f'- 准备耗时约 {prepared["elapsed_seconds"]:.2f} 秒，进程峰值工作集 {prepared["process_peak_working_set_MiB"]:.2f} MiB。',
              f'- 最终验证耗时约 {verified["elapsed_seconds"]:.2f} 秒，进程峰值工作集 {verified["process_peak_working_set_MiB"]:.2f} MiB；这不是整机内存占用。',
              f'- Parquet 合计 {sum(s["bytes"] for s in prepared["partitions"].values()):,} 字节。原始库与旧调试产物只在 .runtime 中，不进入 Git。',
              '- 缓存复用同时核对原始哈希与 staging.py 实现哈希；处理产物签名包含协议、生成代码及 staging 实现。完整旧数据保留在 .runtime/data_archive 作为本次修复前备份，不与当前输出混用。','',
              '## 可复现命令与产物','',
              '从项目根目录执行：','', '```powershell',
              '. .\\scripts\\enter_env.ps1','python -B -m src.data.download',
              'python -B -m src.data.inspect_dataset','python -B -m src.data.prepare',
              'python -B -m src.data.validate','python -B -m src.data.report',
              'python -B -m unittest discover -s tests -v','```','',
              '- data/raw/kuairand_pure/：官方压缩包、校验清单、元数据和原始 CSV。',
              '- data/processed/kuairand_pure/：五个 Parquet、训练词表与完成清单。',
              '- reports/data_audit.json、data_preparation.json、data_validation.json：机器可读证据。',
              '- reports/environment_logs/kuairand_*.txt、all_tests_data_stage.txt：实际命令输出。',
              '- configs/data.yaml：固定协议；需要改变协议时使用新输出目录，不覆盖已完成数据。','',
              '本阶段没有训练双塔、DIN、SASRec，也没有真实去偏收益或强化学习策略收益结论。']
    (ROOT/'reports/data_audit.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Rendered reports/data_audit.md from verified actual artifacts')


if __name__=='__main__':
    main()
