"""CLI argument parsing utilities."""


def parse_args(args):
    """Parse CLI arguments from docopt."""
    architecture = args['--architecture']
    batch_size = int(args['--batch-size'])
    cpu_workers = int(args['--cpu-workers'])
    device = args['--device']
    k_folds = int(args['--k-folds']) if args['--k-folds'] and args['--k-folds'] != 'None' else None
    lr = float(args['--lr'])
    epochs = int(args['--epochs'])
    patience = int(args['--patience']) if args['--patience'] and args['--patience'] != 'None' else None
    subsample = int(args['--subsample-size']) if args.get('--subsample-size') and args['--subsample-size'] != 'None' else None
    return architecture, batch_size, cpu_workers, device, k_folds, lr, epochs, patience, subsample
