import pytest
import dellve.service
import dellve.worker
import gevent.event
import tempfile
import uuid
import multiprocessing as mp
import time

@pytest.fixture
def worker(tmpdir):

    token = str(uuid.uuid1())

    class MockWorker(dellve.worker.WorkerAPI):

        uuid = str(uuid.uuid1())

        started_file = tempfile.mkstemp()[1]
        stopped_file = tempfile.mkstemp()[1]

        @property
        def pidfile(self):
            """Summary

            Returns:
                TYPE: Description
            """
            return '.dellve-test.pid'

        def start(self):
            """Summary

            Returns:
                TYPE: Description
            """
            with open(self.started_file, 'w') as file:
                file.write(self.uuid)
                file.flush()

        def stop(self):
            """Summary

            Returns:
                TYPE: Description
            """
            with open(self.stopped_file, 'w') as file:
                file.write(self.uuid)
                file.flush()

        @property
        def workdir(self):
            """Summary

            Returns:
                TYPE: Description
            """
            return str(tmpdir)

    return MockWorker()

# def test_service_start(worker):
#     service = dellve.service.DELLveService(debug=True, daemon_worker=worker)


# def test_service_stop(service):
#     service = dellve.service.DELLveService(debug=True, daemon_worker=worker)


