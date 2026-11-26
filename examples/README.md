# Esempi e validazione

Questa cartella raccoglie gli scenari di esempio per il simulatore e l'ambiente di validazione contro il sistema Serverledge reale eseguito localmente sull'host.

## Scenari (`examples/scenarios/`)

| File | Descrizione |
| --- | --- |
| `startup.toml` (`.json`, `.yaml`) | Scenario dimostrativo multiregione (`edge`, `regional` e `cloud` astratto `always_succeeds`) in tre formati equivalenti. |
| `validation.toml` | Scenario allineato 1:1 ai 9 worker Serverledge del lab locale (`examples/validation/`), con `cleanup_interval = 1.0` e latenze locali nulle. |
| `autoscaling.toml` | Scenario dimostrativo con autoscaler regionale PPO su due ring eterogenei (`amd64` e `arm64`). |

Per eseguire uno scenario dal simulatore usando il launcher nella root:

```bash
./run.sh examples/scenarios/startup.toml --arrivals 100
./run.sh examples/scenarios/validation.toml --arrivals 100
./run.sh examples/scenarios/autoscaling.toml --arrivals 200
```

## Validazione contro Serverledge reale (`examples/validation/`)

Per avviare il lab locale, eseguire simulatore e sistema reale sulla stessa identica traccia di arrivi/tempi di servizio e confrontare i risultati:

```bash
./examples/validation/validate.sh run --arrivals 100
```

Vedi [examples/validation/README.md](validation/README.md) per tutte le opzioni (incluso `--local-etcd`, `--scenario`, `--down` e i sottocomandi granulari).
