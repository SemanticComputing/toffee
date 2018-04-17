FROM docker.elastic.co/elasticsearch/elasticsearch:5.1.1

USER 0
RUN apk add --no-cache --repository http://dl-3.alpinelinux.org/alpine/edge/testing/ \
    g++ make icu-dev musl-dev wget unzip autoconf libxml++-2.6-dev libarchive-dev \
    automake libtool python3 file

ENV PKG_CONFIG_PATH /usr/local/lib/pkgconfig

ARG HFSTO_VERSION=0.4.5
ARG VOIKKO_VERSION=2.2
ENV HFSTO_VERSION ${HFSTO_VERSION}
ENV VOIKKO_VERSION ${VOIKKO_VERSION}

# Disable x-pack
ENV xpack.security.enabled false
ENV xpack.monitoring.enabled false

ADD * /tmp/

# Install the Voikko plugin and dependencies.
# All of this has to be done in a single RUN because of this bug:
# https://github.com/elastic/elasticsearch-docker/issues/44
# The plugin requires Elasticsearch version 5.1.1, so using to a newer version
# is not possible.
WORKDIR /tmp
RUN wget https://github.com/hfst/hfst-ospell/releases/download/v$HFSTO_VERSION/hfstospell-$HFSTO_VERSION.tar.gz \
    && tar xzvf hfstospell* \
    && cd hfstospell* \
    && ./configure && make && make install \
    && cd .. \
    && wget https://github.com/voikko/corevoikko/archive/rel-voikko-fi-$VOIKKO_VERSION.tar.gz \
    && tar xzvf rel-voikko-fi* \
    && cd corevoikko-rel-voikko-fi-$VOIKKO_VERSION/libvoikko \
    && ./autogen.sh \
    && ./configure --with-dictionary-path=/usr/share/voikko:/usr/lib/voikko \
    && make \
    && make install \
    && mkdir -p /usr/lib/voikko/ \
    && cd /usr/lib/voikko/ \
    && wget http://www.puimula.org/htp/testing/voikko-snapshot-v5/dict-morpho.zip \
    && unzip dict-morpho.zip \
    && elasticsearch-plugin install https://github.com/EvidentSolutions/elasticsearch-analysis-voikko/releases/download/v0.5.0/elasticsearch-analysis-voikko-0.5.0.zip \
    && mv /tmp/.java.policy /usr/share/elasticsearch/ \
    && chown -R elasticsearch:elasticsearch /usr/share/elasticsearch \
    && rm -rf /tmp/*

ENV ES_JAVA_OPTS -Djava.security.policy=file:/usr/share/elasticsearch/.java-policy
USER elasticsearch
WORKDIR /usr/share/elasticsearch

EXPOSE 9200 9300
