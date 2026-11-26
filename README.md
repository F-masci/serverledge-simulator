# Serverledge Simulator

Simulatore Python della piattaforma serverless Serverledge per lo studio delle politiche di autoscaling.

## Avvio rapido

Usa lo script `run.sh` nella root per attivare automaticamente l'ambiente Conda (`serverledge-simulator`, o quello indicato con `--env` / `CONDA_ENV`) e avviare il simulatore:

```bash
./run.sh
./run.sh examples/scenarios/startup.toml --arrivals 100
./run.sh examples/scenarios/autoscaling.toml --arrivals 200 --log-level INFO
./run.sh --env serverledge-simulator examples/scenarios/validation.toml --arrivals 100
```

## Validazione contro Serverledge reale

La cartella [examples/validation/](examples/validation/README.md) contiene lo script `validate.sh` per avviare un cluster Serverledge reale sull'host, eseguire simulatore e sistema reale sulla stessa traccia e confrontarne le metriche:

```bash
./examples/validation/validate.sh run --arrivals 100
```
