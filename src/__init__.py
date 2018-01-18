import eventlet; eventlet.monkey_patch() # noqa
import logging

log = logging.getLogger(__name__)
log.info('Package init: {}'.format(__name__))
