# Evidências: agendamento nativo do GitHub e contingência automática

Este diretório documenta por que os ciclos avaliados foram disparados pela **sequência automática**
(`workflow_dispatch` com `origem = inicio/encadeado`) e não pelo gatilho `schedule`, apesar de o cron estar
configurado no workflow desde o primeiro commit.

## Linha do tempo (23/11/2026, horários UTC; Brasília = UTC−3)

| Horário (UTC) | Evento | Evidência |
|---|---|---|
| 12:11 | Primeiro commit já com `schedule` (`cron: "17 9,21 * * *"`) | `historico_cron_commits.txt`, `historico_cron_valores.txt` |
| 12:12 e 12:16 | Dois ciclos de validação manuais: sucesso; depois o estado foi zerado | `api_runs_trabalho.json` |
| 12:35 | Cron alterado para `"17 13-19 * * *"` (de hora em hora) | idem |
| 13:17, 14:17, 15:17 | Horários do cron sem nenhuma execução | `api_runs_schedule_trabalho.json` |
| 14:22 | Workflow desligado e religado para forçar novo registro do cron | — |
| 15:44 | Cron alterado para `"7,27,47 * * * *"` (a cada 20 min) | `historico_cron_*` |
| 15:47 a 19:27 | Horários do cron sem nenhuma execução | `api_runs_schedule_trabalho.json` |
| 17:43 | Repositório de teste mínimo criado: workflow que só imprime uma mensagem, `cron: "*/5 * * * *"` | `repo_teste_workflow.yml`, `repo_teste_info.json` |
| 17:44 | Branch padrão trocada e restaurada para forçar novo registro do cron | — |
| 17:45 a 18:13 | Repositório de teste: 6 horários de 5 min sem nenhuma execução | `api_runs_schedule_repo_teste.json` |
| 18:15 | Sequência automática iniciada (`origem = inicio`, 6 ciclos encadeados, intervalo de 5 min) | `../../evidencias/ciclos/*/execucao.json` |
| 18:15 a 18:58 | 7 ciclos concluídos com sucesso, sem intervenção | `api_runs_trabalho.json`, `../prints/01_actions_execucoes.png` |
| 19:30 | Coleta final: ainda 0 execuções por `schedule` nos dois repositórios | `api_runs_schedule_*.json`, `consultado_em.txt` |

Em nenhum momento a página de status do GitHub indicou incidente no Actions (`../prints/05_github_status.png`).

## Conclusão

O mesmo arquivo de workflow executa com sucesso quando disparado, e um workflow mínimo agendado a cada 5 minutos
também não foi executado em outro repositório da mesma conta. Logo, a falha está no serviço de agendamento do GitHub
para esta conta, não na configuração do projeto. A documentação do GitHub informa que execuções `schedule` podem ser
atrasadas ou descartadas sem aviso.

## Como conferir

```bash
gh run list --repo Bornety0/projeto-n3-mlops --event schedule
gh run list --repo Bornety0/projeto-n3-mlops --workflow pipeline_mlops.yml
git log --date=iso --format='%h %ad %s' -- .github/workflows/pipeline_mlops.yml
```

Na interface: *Actions → pipeline-mlops*, filtro `event:schedule` (`../prints/02_actions_filtro_schedule_vazio.png`).

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `consultado_em.txt` | Momento da coleta dos dados abaixo |
| `api_runs_schedule_trabalho.json` | Total de execuções por `schedule` no repositório do trabalho (API) |
| `api_runs_schedule_repo_teste.json` | Total de execuções por `schedule` no repositório de teste (API) |
| `api_runs_trabalho.json` | Todas as execuções do workflow do trabalho, com gatilho e resultado |
| `historico_cron_commits.txt` / `historico_cron_valores.txt` | Commits que alteraram o workflow e o cron vigente em cada um |
| `repo_teste_workflow.yml` / `repo_teste_info.json` | Workflow mínimo do repositório de teste e datas de criação/push |
