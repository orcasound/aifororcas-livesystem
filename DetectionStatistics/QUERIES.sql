-- #1: query total number of detections (audio time required to moderate)
SELECT count(1) 
from c
where c.timestamp like '%2022%'
-- RESULT
-- total ~5.5k, 92hrs


-- #2: more granular breakdown of detections
select count(1) as num_candidates, c.reviewed, c.SRKWFound from c 
group by c.reviewed, c.SRKWFound
-- RESULT
-- not moderated ~1.8k, 30hrs
-- moderated ~3.7k, 62hrs (9/52/0.5 yes/no/dunno)
-- {
--     1819,           (~30 hrs unmoderated)
--     552, 29, 3134   (~62 hrs moderated: 9, 0.5, 52 hrs yes/dunno/no)
-- }


-- #3: Aggregate individual detections into number of events
-- for reference, spreadsheet of known events (sheet: "summary stats") 
-- https://docs.google.com/spreadsheets/d/1Js1CgbmK0Vbe3m0DfiFim1BE4lXMzC75S7GN-7QEE7Y/edit#gid=0


-- #4: start date and approx runtime of each hydrophone
select min(c.timestamp) as start_time, c.source_guid as location
from c
group by c.source_guid
-- RESULT
-- total runtime ~2.25k days, ~54k hrs (per hydrophone rpi_port_townsend/rpi_bush_point/rpi_orcasound_lab)
-- [
--     {
--         "location": "rpi_port_townsend",
--         "start_date": "2020-11-17",
--         "days_since_start": 733
--     },
--     {
--         "location": "rpi_bush_point",
--         "start_date": "2020-11-17",
--         "runtime": 733
--     },
--     {
--         "location": "rpi_orcasound_lab",
--         "start_date": "2020-09-28",
--         "runtime": 783
--     }
-- ]
