
import dellve.worker
import gevent
import gevent.event

def test_worker():
    # Create worker
    worker = dellve.worker.Worker(port=6666)

    # Start worker in a separate thread
    worker_greenlet = gevent.Greenlet.spawn(dellve.worker.Worker.start, worker)

    gevent.sleep(2.5) # give worker some time to start

    # Worker must be running
    assert not worker_greenlet.ready()

    gevent.sleep(2.5) # wait a little...

    # Worker must still be running
    assert not worker_greenlet.ready()

    # Stop worker
    worker.stop()

    gevent.sleep(2.5) # give worker some time to stop

    # Worker must not be running
    assert worker_greenlet.ready()
