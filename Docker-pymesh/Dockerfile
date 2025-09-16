FROM pymesh/pymesh:latest

RUN mkdir Data && mkdir Code

COPY init.sh /usr/local/bin/init.sh
RUN chmod +x /usr/local/bin/init.sh
ENTRYPOINT ["/usr/local/bin/init.sh"]