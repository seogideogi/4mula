
CREATE TABLE dps_fraud_tab AS
SELECT 
    dps_fc_ac,
    tran_dt,
    tran_tmrg,
    ff_sp_ai,
    LAG(COUNT(*)) OVER (PARTITION BY dps_fc_ac ORDER BY tran_dt ) AS prev_dps_fraud_cnt  -- 전일자의 Count
FROM HF_TRNS_TRAN hf
WHERE ff_sp_ai = 'SP'
GROUP BY dps_fc_ac, tran_dt, tran_tmrg
ORDER BY tran_dt, tran_tmrg, dps_fc_ac;


CREATE TABLE wd_fraud_tab AS
SELECT 
    wd_fc_ac,
    tran_dt,
    tran_tmrg,
    ff_sp_ai,
    LAG(COUNT(*)) OVER (PARTITION BY wd_fc_ac ORDER BY tran_dt ) AS prev_wd_fraud_cnt  -- 전일자의 Count
FROM HF_TRNS_TRAN hf
WHERE ff_sp_ai = 'SP'
GROUP BY wd_fc_ac, tran_dt, tran_tmrg
ORDER BY tran_dt, tran_tmrg, wd_fc_ac;







-- cnt
SELECT 
    COUNT(*)
FROM HF_TRNS_TRAN AS hf
INNER JOIN dps_fnd AS df
    ON hf.dps_fc_ac = df.dps_fc_ac
    AND hf.tran_dt = df.tran_dt
    AND hf.tran_tmrg = df.tran_tmrg
    AND hf.fnd_type = df.fnd_type
INNER JOIN dps_md AS dm
    ON hf.dps_fc_ac = dm.dps_fc_ac
    AND hf.tran_dt = dm.tran_dt
    AND hf.tran_tmrg = dm.tran_tmrg
    AND hf.md_type = dm.md_type
INNER JOIN wd_fnd AS wf
    ON hf.wd_fc_ac = wf.wd_fc_ac
    AND hf.tran_dt = wf.tran_dt
    AND hf.tran_tmrg = wf.tran_tmrg
    AND hf.fnd_type = wf.fnd_type
INNER JOIN wd_md AS wm
    ON hf.wd_fc_ac = wm.wd_fc_ac
    AND hf.tran_dt = wm.tran_dt
    AND hf.tran_tmrg = wm.tran_tmrg
    AND hf.md_type = wm.md_type
LEFT JOIN dps_fraud_tab AS dps_frd
    ON hf.dps_fc_ac = dps_frd.dps_fc_ac
    AND hf.tran_dt = dps_frd.tran_dt
    AND hf.tran_tmrg = dps_frd.tran_tmrg
    AND hf.ff_sp_ai = dps_frd.ff_sp_ai
LEFT JOIN wd_fraud_tab AS wd_frd
    ON hf.wd_fc_ac = wd_frd.wd_fc_ac
    AND hf.tran_dt = wd_frd.tran_dt
    AND hf.tran_tmrg = wd_frd.tran_tmrg
    AND hf.ff_sp_ai = wd_frd.ff_sp_ai;
------- 9947307

CREATE TABLE HF_TRNS_TRAN_tmp_1
AS 
SELECT 
	hf.wd_fc_ac
	, hf.dps_fc_ac
	, hf.tran_dt
	, hf.tran_tmrg
    , hf.tran_amt
	, hf.md_type
	, hf.fnd_type
	, df.dps_fc_ac_tran_amt AS dps_fc_ac_fnd_amt
	, df.dps_fc_ac_cnt AS dps_fc_ac_fnd_cnt
	, dm.dps_fc_ac_tran_amt AS  dps_fc_ac_md_amt
	, dm.dps_fc_ac_cnt AS dps_fc_ac_md_cnt
	, wf.wd_fc_ac_tran_amt AS wd_fc_ac_fnd_amt
	, wf.wd_fc_ac_cnt AS wd_fc_ac_fnd_cnt
	, wm.wd_fc_ac_tran_amt AS wd_fc_ac_md_amt
	, wm.wd_fc_ac_cnt AS wd_fc_ac_md_cnt
	, dps_frd.prev_dps_fraud_cnt
	, wd_frd.prev_wd_fraud_cnt
	, hf.ff_sp_ai
FROM HF_TRNS_TRAN AS hf
INNER JOIN dps_fnd AS df
    ON hf.dps_fc_ac = df.dps_fc_ac
    AND hf.tran_dt = df.tran_dt
    AND hf.tran_tmrg = df.tran_tmrg
    AND hf.fnd_type = df.fnd_type
INNER JOIN dps_md AS dm
    ON hf.dps_fc_ac = dm.dps_fc_ac
    AND hf.tran_dt = dm.tran_dt
    AND hf.tran_tmrg = dm.tran_tmrg
    AND hf.md_type = dm.md_type
INNER JOIN wd_fnd AS wf
    ON hf.wd_fc_ac = wf.wd_fc_ac
    AND hf.tran_dt = wf.tran_dt
    AND hf.tran_tmrg = wf.tran_tmrg
    AND hf.fnd_type = wf.fnd_type
INNER JOIN wd_md AS wm
    ON hf.wd_fc_ac = wm.wd_fc_ac
    AND hf.tran_dt = wm.tran_dt
    AND hf.tran_tmrg = wm.tran_tmrg
    AND hf.md_type = wm.md_type
LEFT JOIN dps_fraud_tab AS dps_frd
    ON hf.dps_fc_ac = dps_frd.dps_fc_ac
    AND hf.tran_dt = dps_frd.tran_dt
    AND hf.tran_tmrg = dps_frd.tran_tmrg
    AND hf.ff_sp_ai = dps_frd.ff_sp_ai
LEFT JOIN wd_fraud_tab AS wd_frd
    ON hf.wd_fc_ac = wd_frd.wd_fc_ac
    AND hf.tran_dt = wd_frd.tran_dt
    AND hf.tran_tmrg = wd_frd.tran_tmrg
    AND hf.ff_sp_ai = wd_frd.ff_sp_ai;
	
	
	
	
-- dps




create table dps_fnd
as
select 
    dps_fc_ac
    , tran_dt
    , tran_tmrg
    , fnd_type
    , sum(tran_amt) as dps_fc_ac_tran_amt
    , count(*) dps_fc_ac_cnt
from HF_TRNS_TRAN hf
group by dps_fc_ac, tran_dt, tran_tmrg, fnd_type
order by tran_dt, tran_tmrg, dps_fc_ac, fnd_type
;

create table dps_md
as
select 
    dps_fc_ac
    , tran_dt
    , tran_tmrg
    , md_type
    , sum(tran_amt) as dps_fc_ac_tran_amt
    , count(*) dps_fc_ac_cnt
from HF_TRNS_TRAN hf
group by dps_fc_ac, tran_dt, tran_tmrg, md_type
order by tran_dt, tran_tmrg, dps_fc_ac, md_type
;

-- wd
create table wd_md
as
select 
    wd_fc_ac
    , tran_dt
    , tran_tmrg
    , md_type
    , sum(tran_amt) as wd_fc_ac_tran_amt
    , count(*) wd_fc_ac_cnt
from HF_TRNS_TRAN hf
group by wd_fc_ac, tran_dt, tran_tmrg, md_type
order by tran_dt, tran_tmrg, wd_fc_ac, md_type
;

create table wd_fnd
as
select 
    wd_fc_ac
    , tran_dt
    , tran_tmrg
    , fnd_type
    , sum(tran_amt) as wd_fc_ac_tran_amt
    , count(*) wd_fc_ac_cnt
from HF_TRNS_TRAN hf
group by wd_fc_ac, tran_dt, tran_tmrg, fnd_type
order by tran_dt, tran_tmrg, wd_fc_ac, fnd_type
;


CREATE TABLE dps_hf_trns_tran 
AS
WITH base_data AS (
    SELECT
        tran_dt,
        tran_tmrg,
        dps_fc_ac,
        tran_amt,
        md_type,
        fnd_type,
        ff_sp_ai,
        STR_TO_DATE(CONCAT(STR_TO_DATE(CAST(tran_dt AS CHAR), '%Y%m%d'), ' ', tran_tmrg), '%Y-%m-%d %H:%i:%s') AS tran_datetmrg
    FROM
        HF_TRNS_TRAN
),
time_ranges AS (
    SELECT 3 AS hours 
    UNION ALL 
    SELECT 12 AS hours 
    UNION ALL 
    SELECT 24 AS hours 
    UNION ALL 
    SELECT 24 * 7 AS hours 
    UNION ALL 
    SELECT 24 * 30 AS hours 
    UNION ALL 
    SELECT 24 * 90 AS hours 
),
aggregated_data AS (
    SELECT
        b1.tran_dt,
        b1.tran_tmrg,
        b1.dps_fc_ac,
        COUNT(b2.tran_amt) AS tran_cnt,
        SUM(b2.tran_amt) AS tran_amt,
        tr.hours  -- hours 컬럼을 이곳에서 SELECT해야 사용 가능
    FROM
        base_data b1
    CROSS JOIN
        time_ranges tr  -- 시간 범위 테이블 Cross Join
    LEFT JOIN
        base_data b2 ON b1.dps_fc_ac = b2.dps_fc_ac
                       AND b2.tran_datetmrg >= DATE_SUB(b1.tran_datetmrg, INTERVAL tr.hours HOUR)
                       AND b2.tran_datetmrg < b1.tran_datetmrg
    GROUP BY
        b1.tran_dt,
        b1.tran_tmrg,
        b1.dps_fc_ac,
        tr.hours  -- 여기서 GROUP BY에 hours 추가
),
dps_HF_TRNS_TRAN AS (
    SELECT
        tran_dt,
        tran_tmrg,
        dps_fc_ac,
        MAX(CASE WHEN hours = 3 THEN tran_cnt END) AS dps_fc_ac_3H_TRAN_CNT,
        MAX(CASE WHEN hours = 12 THEN tran_cnt END) AS dps_fc_ac_12H_TRAN_CNT,
        MAX(CASE WHEN hours = 24 THEN tran_cnt END) AS dps_fc_ac_1D_TRAN_CNT,
        MAX(CASE WHEN hours = 24 * 7 THEN tran_cnt END) AS dps_fc_ac_7D_TRAN_CNT,
        MAX(CASE WHEN hours = 24 * 30 THEN tran_cnt END) AS dps_fc_ac_30D_TRAN_CNT,
        MAX(CASE WHEN hours = 24 * 90 THEN tran_cnt END) AS dps_fc_ac_90D_TRAN_CNT,
        MAX(CASE WHEN hours = 3 THEN tran_amt END) AS dps_fc_ac_3H_TRAN_AMT,
        MAX(CASE WHEN hours = 12 THEN tran_amt END) AS dps_fc_ac_12H_TRAN_AMT,
        MAX(CASE WHEN hours = 24 THEN tran_amt END) AS dps_fc_ac_1D_TRAN_AMT,
        MAX(CASE WHEN hours = 24 * 7 THEN tran_amt END) AS dps_fc_ac_7D_TRAN_AMT,
        MAX(CASE WHEN hours = 24 * 30 THEN tran_amt END) AS dps_fc_ac_30D_TRAN_AMT,
        MAX(CASE WHEN hours = 24 * 90 THEN tran_amt END) AS dps_fc_ac_90D_TRAN_AMT
    FROM
        aggregated_data
    GROUP BY
        tran_dt,
        tran_tmrg,
        dps_fc_ac
)
SELECT * FROM dps_HF_TRNS_TRAN;





CREATE TABLE WD_HF_TRNS_TRAN
AS 
WITH base_data AS (
    SELECT
        tran_dt,
        tran_tmrg,
        wd_fc_ac,
        tran_amt,
        md_type,
        fnd_type,
        ff_sp_ai,
        STR_TO_DATE(CONCAT(STR_TO_DATE(CAST(tran_dt AS CHAR), '%Y%m%d'), ' ', tran_tmrg), '%Y-%m-%d %H:%i:%s') AS tran_datetmrg
    FROM
        HF_TRNS_TRAN
),
time_ranges AS (
    SELECT 3 AS hours 
    UNION ALL 
    SELECT 12 AS hours 
    UNION ALL 
    SELECT 24 AS hours 
    UNION ALL 
    SELECT 24 * 7 AS hours 
    UNION ALL 
    SELECT 24 * 30 AS hours 
    UNION ALL 
    SELECT 24 * 90 AS hours 
),
aggregated_data AS (
    SELECT
        b1.tran_dt,
        b1.tran_tmrg,
        b1.wd_fc_ac,
        COUNT(b2.tran_amt) AS tran_cnt,
        SUM(b2.tran_amt) AS tran_amt,
        tr.hours  -- hours 컬럼을 이곳에서 SELECT해야 사용 가능
    FROM
        base_data b1
    CROSS JOIN
        time_ranges tr  -- 시간 범위 테이블 Cross Join
    LEFT JOIN
        base_data b2 ON b1.wd_fc_ac = b2.wd_fc_ac
                       AND b2.tran_datetmrg >= DATE_SUB(b1.tran_datetmrg, INTERVAL tr.hours HOUR)
                       AND b2.tran_datetmrg < b1.tran_datetmrg
    GROUP BY
        b1.tran_dt,
        b1.tran_tmrg,
        b1.wd_fc_ac,
        tr.hours  -- 여기서 GROUP BY에 hours 추가
),
WD_HF_TRNS_TRAN AS (
    SELECT
        tran_dt,
        tran_tmrg,
        wd_fc_ac,
        MAX(CASE WHEN hours = 3 THEN tran_cnt END) AS WD_FC_AC_3H_TRAN_CNT,
        MAX(CASE WHEN hours = 12 THEN tran_cnt END) AS WD_FC_AC_12H_TRAN_CNT,
        MAX(CASE WHEN hours = 24 THEN tran_cnt END) AS WD_FC_AC_1D_TRAN_CNT,
        MAX(CASE WHEN hours = 24 * 7 THEN tran_cnt END) AS WD_FC_AC_7D_TRAN_CNT,
        MAX(CASE WHEN hours = 24 * 30 THEN tran_cnt END) AS WD_FC_AC_30D_TRAN_CNT,
        MAX(CASE WHEN hours = 24 * 90 THEN tran_cnt END) AS WD_FC_AC_90D_TRAN_CNT,
        MAX(CASE WHEN hours = 3 THEN tran_amt END) AS WD_FC_AC_3H_TRAN_AMT,
        MAX(CASE WHEN hours = 12 THEN tran_amt END) AS WD_FC_AC_12H_TRAN_AMT,
        MAX(CASE WHEN hours = 24 THEN tran_amt END) AS WD_FC_AC_1D_TRAN_AMT,
        MAX(CASE WHEN hours = 24 * 7 THEN tran_amt END) AS WD_FC_AC_7D_TRAN_AMT,
        MAX(CASE WHEN hours = 24 * 30 THEN tran_amt END) AS WD_FC_AC_30D_TRAN_AMT,
        MAX(CASE WHEN hours = 24 * 90 THEN tran_amt END) AS WD_FC_AC_90D_TRAN_AMT
    FROM
        aggregated_data
    GROUP BY
        tran_dt,
        tran_tmrg,
        wd_fc_ac
)
SELECT * FROM WD_HF_TRNS_TRAN;
