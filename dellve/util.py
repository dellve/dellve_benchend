import click
import config
import logging
import multiprocessing as mp
import requests
import sys

# DELLve logger
logger = logging.getLogger('dellve-logger')


def api_url(url, *args, **kwargs):
    host = kwargs.setdefault('host', config.get('http-host'))
    port = kwargs.setdefault('port', config.get('http-port'))
    return 'http://{host}:{port}/{url}'.\
        format(host=host, port=port, url=url) % args


def api_get(url, *args, **kwargs):
    # Get arguments
    code = kwargs.setdefault('data', 200)
    err_msg = kwargs.setdefault('err_msg', None)
    # Get url
    url = api_url(url, *args)

    # Create error message
    if err_msg is not None:
        m = err_msg + ', GET %s FAILED' % url
    else:
        m = 'GET %s FAILED' % url

    try: # Make request
        res = requests.get(url)
    except:
        raise click.ClickException(m)
    else:
        if res.status_code == code:
            return res
        if err_msg:
            logger.error(err_msg)
        raise click.ClickException(m)


def api_post(url, *args, **kwargs):
    # Get arguments
    code = kwargs.setdefault('code', 200)
    data = kwargs.setdefault('data', {})
    err_msg = kwargs.setdefault('err_msg', None)
    # Get url
    url = api_url(url, *args)

    # Create error message
    if err_msg is not None:
        m = err_msg + ', POST %s FAILED' % url
    else:
        m = 'POST %s FAILED' % url

    try: # Make request
        res = requests.post(url, data=data)
    except:
        raise click.ClickException(m)
    else:
        if res.status_code == code:
            return res
        if err_msg:
            logger.error(err_msg)
        raise click.ClickException(m)


class ClickLoggingHandler(logging.Handler):

    def emit(self, rec):
        # If the following is True, we are in main process
        if type(mp.current_process()) != mp.Process:
            click.echo(self.format(rec), err=(rec.levelno >= logging.ERROR))
