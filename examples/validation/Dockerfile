FROM golang:1.24-bookworm AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN make serverledge serverledge-cli lb

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --no-create-home --shell /usr/sbin/nologin serverledge
COPY --from=build /src/bin/serverledge /src/bin/serverledge-cli /src/bin/lb /usr/local/bin/
USER serverledge
