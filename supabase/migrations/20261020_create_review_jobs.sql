create table if not exists public.review_jobs (
  id uuid primary key,
  repo text not null,
  pr_number integer not null check (pr_number > 0),
  endpoint text,
  status text not null,
  payload jsonb not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists review_jobs_status_idx on public.review_jobs (status);

alter table public.review_jobs enable row level security;
revoke all on public.review_jobs from anon, authenticated;
grant all on public.review_jobs to service_role;
