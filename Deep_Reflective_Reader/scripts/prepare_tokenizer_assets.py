"""Explicit setup-only acquisition; --verify-only never accesses the network."""

import argparse
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from time import monotonic
from urllib.request import urlopen

from llm.offline_token_counter import (
    OfflineTokenCounter, RECIPE_VERSION, SUPPORTED_MODELS, encoding_asset_for_model,
)

IO_TIMEOUT_SECONDS = 15
DOWNLOAD_DEADLINE_SECONDS = 180


class AssetSetupError(Exception):
    pass


def required_assets():
    assets = {}
    for model in SUPPORTED_MODELS:
        asset = encoding_asset_for_model(model)
        if asset is None:
            raise AssetSetupError("unsupported_model_mapping")
        assets[asset.name] = asset
    return tuple(assets.values())


def verified_file(path, asset):
    if not path.is_file():
        return False
    with path.open("rb") as stream:
        data = stream.read(asset.size + 1)
    return len(data) == asset.size and hashlib.sha256(data).hexdigest() == asset.sha256


def install_asset(directory, asset):
    destination = directory / asset.filename
    if verified_file(destination, asset):
        return
    print(f"Preparing tokenizer asset: encoding={asset.name} expected_bytes={asset.size}", flush=True)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = None
    started = monotonic()
    try:
        # Stage beside the destination so replacement is atomic on this filesystem.
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".tokenizer-", delete=False) as output:
            temporary = Path(output.name)
            size = 0
            with urlopen(asset.url, timeout=IO_TIMEOUT_SECONDS) as response:
                while True:
                    if monotonic() - started > DOWNLOAD_DEADLINE_SECONDS:
                        raise AssetSetupError("download_deadline_exceeded")
                    chunk = response.read1(min(65536, asset.size + 1 - size))
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > asset.size:
                        raise AssetSetupError("download_size_exceeded")
                    output.write(chunk)
            if monotonic() - started > DOWNLOAD_DEADLINE_SECONDS:
                raise AssetSetupError("download_deadline_exceeded")
            output.flush()
            os.fsync(output.fileno())
        if not verified_file(temporary, asset):
            raise AssetSetupError(f"download_integrity_failed:expected_bytes={asset.size}:received_bytes={size}")
        temporary.chmod(0o644)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def verify_assets(directory):
    for asset in required_assets():
        if not verified_file(directory / asset.filename, asset):
            raise AssetSetupError("asset_missing_or_corrupt")
    counter = OfflineTokenCounter(directory)
    for model in SUPPORTED_MODELS:
        count = counter.count(model, "Tokenizer startup check.")
        if not isinstance(count, int) or count <= 0:
            raise AssetSetupError("runtime_tokenizer_unavailable")
    return counter.identity(SUPPORTED_MODELS[0])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path, default=OfflineTokenCounter().asset_dir)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not args.verify_only:
            for asset in required_assets():
                install_asset(args.asset_dir, asset)
        identity = verify_assets(args.asset_dir)
    except (AssetSetupError, OSError, ValueError) as error:
        reason = str(error) if isinstance(error, AssetSetupError) else type(error).__name__
        print(f"Tokenizer setup failed: {reason}. Run python -m scripts.prepare_tokenizer_assets "
              "in the backend environment; startup verification never downloads assets.", file=sys.stderr)
        return 1
    print(f"Tokenizer assets verified: recipe={RECIPE_VERSION} identity={identity}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
