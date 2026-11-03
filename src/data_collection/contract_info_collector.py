import os
import re
import json
import pandas
import subprocess
import pytz
from tqdm import tqdm
from dateutil import parser
import time
import networkx as nx


class ContractInfoCollector:

    def __init__(self, cache_path='./data/cache/contract_info'):
        self.cache_path = cache_path
        os.makedirs(cache_path, exist_ok=True)

    @staticmethod
    def _fetch_website_content(url):
        try:
            result = subprocess.run(['curl', '-s', url],
                                    capture_output=True,
                                    text=True,
                                    check=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            # print(f"Error fetching website: {e}")
            return None

    @staticmethod
    def _find_and_convert_datetimes(content):
        pattern = r'([A-Z][a-z]{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2} [AP]M UTC)'
        datetime_strings = re.findall(pattern, content)
        if not datetime_strings:
            return ''
        local_tz = pytz.timezone(time.tzname[0])
        converted_datetimes = []
        for dt_str in datetime_strings:
            dt_utc = parser.parse(dt_str)
            dt_local = dt_utc.replace(tzinfo=pytz.UTC).astimezone(local_tz)
            converted_datetimes.append({
                'original': dt_str,
                'local': dt_local.strftime('%Y-%m-%d %H:%M:%S %Z')
            })
        return converted_datetimes[0]

    def _get_contract_address_link(self, address, chain):
        if 'eth' in self.txn_link:
            self.contract_link = 'https://etherscan.io/address/{}'.format(address)
        elif 'bsc' in self.txn_link:
            self.contract_link = 'https://bscscan.com/address/{}'.format(address)
        elif 'polygon' in self.txn_link:
            self.contract_link = 'https://polygonscan.com/address/{}'.format(address)
        elif 'optimism' in self.txn_link:
            self.contract_link = 'https://optimistic.etherscan.com/address/{}'.format(address)
        elif 'arbitrum' in self.txn_link:
            self.contract_link = 'https://arbiscan.io/address/{}'.format(address)
        elif 'avalanche' in self.txn_link:
            self.contract_link = 'https://snowtrace.io/address/{}'.format(address)
        elif 'fantom' in self.txn_link:
            self.contract_link = 'https://ftmscan.com/address/{}'.format(address)
        elif 'gnosis' in self.txn_link:
            self.contract_link = 'https://gnosisscan.io/address/{}'.format(address)
        elif 'base' in self.txn_link:
            self.contract_link = 'https://basescan.org/address/{}'.format(address)

    def _is_contract(self, content):
        if 'Contract Creator' in content:
            return True
        return False

    def _get_contract_creator(self, content):
        pass

    def _check_cache(self, txn_hash):
        txn_hash = txn_hash.lower()
        cache_file = os.path.join(self.cache_path, f"{txn_hash}.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
        return None

    def _save_cache(self, txn_hash, data):
        txn_hash = txn_hash.lower()
        cache_file = os.path.join(self.cache_path, f"{txn_hash}.json")
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)

    def run(self, address, chain):
        self._get_contract_address_link(address, chain)
        if self.contract_link:
            content = self._fetch_website_content(self.contract_link)
            if content:
                is_contract = self._is_contract(content)
                if is_contract:
                    creator = self._get_contract_creator(content)
                    result = {
                        "contract_address": address,
                        "creator": creator,
                        "is_contract": is_contract
                    }
                    self._save_cache(address, result)
                    return result
                else:
                    return {'is_contract': False}
        return {}


if __name__ == "__main__":
    # df = pandas.read_csv('./transaction_links.csv')
    # df = df.dropna(subset=['txn_link'])
    # df['txn_link'] = df['txn_link'].apply(lambda x: eval(x))
    # for txn_links in tqdm(df['txn_link'].to_list()[868:]):
    #     for txn_link in txn_links:
    #         TransactionInfoCollector(txn_link).run()
    # TransactionInfoCollector('').revise_token_transferred()

    df = pandas.read_csv('./dumps/transaction_links_defi.csv')
    df = df.dropna(subset=['attack_tx'])
    df['attack_tx'] = df['attack_tx'].apply(eval)
    txn_links_list = df['attack_tx'].tolist()
    for txn_links in tqdm(txn_links_list):
        for txn_link in txn_links:
            ContractInfoCollector(txn_link).run()
