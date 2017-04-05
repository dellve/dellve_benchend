import api
import benchmark
import click
import daemon
import config
import os
import pick
import template
import time
import tqdm

@click.group()
@click.option('--config-file', 'config_file',
    help='Configuration file name.', type=click.File('r'))
def cli(config_file):
    """DELLve benchmark command line interface.

    Type 'dellve COMMAND --help' to see help for commands listed below.
    """
    if config_file:
        config.load(config_file) # load DELLve configuration

@cli.command('ls', short_help='List installed benchmarks.')
def ls():
    """Lists installed DELLve benchmarks.
    """
    for benchmark in config.get('benchmarks'):
        print ' ', benchmark.name

@cli.command('start', short_help='Start the benchmark service.')
@click.option('--debug', 'debug', default=False,
    help='Turn on debug mode.', is_flag=True)
def start(debug):
    """Starts DELLve benchmark background service.
    """
    click.echo('Starting benchmark service...')
    daemon.Daemon(debug=debug).do_action('start')

@cli.command('status', short_help='Get the status of benchmark service.')
def status():
    """Gets the status DELLve benchmark background service.
    """
    click.echo('Getting benchmark status...')
    daemon.Daemon().do_action('status')

@cli.command('stop', short_help='Stop the benchmark service.')
def stop():
    """Stops DELLve benchmark background service.
    """
    click.echo('Stopping benchmark service...')
    daemon.Daemon().do_action('stop')

@cli.command('run', short_help='Run the benchmarks.')
@click.option('--all', '-A', 'run_all', default=False, is_flag=True,
    help='Run all benchmarks.')
def run(run_all):
    """Runs the user specified benchmarks
    """

    # Load benchmarks
    benchmarks=list(config.get('benchmarks'))

    if len(benchmarks) < 1: # ensure there're some benchmarks to be run
        click.echo('Please, install at least one benchmark plugin.', err=True)
        return

    # Note: user may select to run all benchmarks with --all/-A flag;
    #       if -all/-A flag is set, we run every single one of the 'benchmarks';
    #       otherwise, we prompt user to select a subset of the 'benchmarks'.

    if not run_all:
        options = [b.name for b in benchmarks]
        title = '\n'.join([
            'Please select benchmarks to run:',
            '',
            'Press UP and DOWN arrow keys to navigate',
            'Press SPACE to select benchmarks',
            'Press ENTER to proceede'
        ])

        # Ask user to pick benchmarks that he/she/ze wants to run
        picked = pick.pick(options, title, indicator='+', multi_select=True)

        # Note: pick() returns list of (name, index) pairs; since we only care
        #       about names in this list, we remove indexes from it below
        selected = {name for name, index in picked}

        # Filter out benchmarks that weren't selected
        benchmarks = [b for b in benchmarks if b.name in selected]

    for benchmark_class in benchmarks:
        with tqdm.tqdm(desc=benchmark_class.name, total=100) as progress_bar:
            benchmark = benchmark_class.load()()
            benchmark.start()
            old_progress = 0
            while benchmark.is_running():
                new_progress = benchmark.progress
                progress_bar.update(new_progress - old_progress)
                old_progress = new_progress
                time.sleep(0.01)


@cli.command('new', short_help='Create a new benchmark.')
def new():
    """Generates the boilerplate code for a new DELLve-benchmark plugin
    """
    dir_name = click.prompt('Please enter a directory name', default='./')
    package_name = click.prompt('Please enter a Python package name')
    benchmark_name = click.prompt('Please enter a unique benchmark class name')
    template.Template().render(dir_name, package_name, benchmark_name)
