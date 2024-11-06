select count(*) ,'general' as "bug_type"
from commitpackft_classified
where bug_type like '%general%'
union 
select count(*), 'mobile' as "bug_type" 
from commitpackft_classified_train
where bug_type like '%mobile%'
union
select count(*), 'functionality' as "bug_type" 
from commitpackft_classified_train
where bug_type like '%functionality%'
union
select count(*), 'ui-ux' as "bug_type" 
from commitpackft_classified_train
where bug_type like '%ui-ux%'
union
select count(*), 'compatibility-performance' as "bug_type" 
from commitpackft_classified_train
where bug_type like '%compatibility-performance%'
union
select count(*), 'network-security' as "bug_type" 
from commitpackft_classified_train
where bug_type like '%network-security%';