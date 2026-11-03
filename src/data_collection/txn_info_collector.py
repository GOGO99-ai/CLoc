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
import requests
class TransactionInfoCollector:

    def __init__(self):
        pass
    
    @staticmethod
    def _fetch_website_content(url):
        try:
            result = subprocess.run(['curl', '-s', url],
                                    capture_output=True,
                                    text=True,
                                    check=True)
            #print("Info",result.stdout)
            return result.stdout
        except subprocess.CalledProcessError as e:
            # print(f"Error fetching website: {e}")
            return None

    @staticmethod
    def _find_and_convert_datetimes(content):
        try:
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
        except Exception as e:
            return ''
    
    @staticmethod
    def _get_transaction_status(content):
        if '<!-- Status -->' in content:
            return content.split('<!-- Status -->')[1].split('</span></div>')[0].split('</i>')[-1].strip()
        else:
            return 'unknown'
    
    @staticmethod
    def _get_transaction_gas(content):
        pass
        
        
    @staticmethod
    def _get_transaction_block(content):
        block_number = -1
        confirmations = -1
        if '<!-- Block Height -->' in content:
            block_number = content.split('<!-- Block Height -->')[1].split('</a></span>')[0].split('>')[-1].strip()
            if 'Block Confirmations' in content:
                confirmations = content.split('Block Confirmations')[0].split('>')[-1].strip()
        else:
            pass
        return {
            'block_number': block_number,
            'confirmations': confirmations
        }

    @staticmethod
    def get_transaction_hash(txn_link):
        pattern = r'(0x[0-9a-fA-F]{64})'
        return re.findall(pattern, txn_link)
    
   
    def get_transaction_link(self, keyword):
        if 'eth' in keyword:
            self.txn_link = 'https://etherscan.io/tx/{}'.format(self.txn_hash)
        elif 'bsc' in keyword:
            self.txn_link = 'https://bscscan.com/tx/{}'.format(self.txn_hash)
        elif 'polygon' in keyword:
            self.txn_link = 'https://polygonscan.com/tx/{}'.format(self.txn_hash)
        elif 'optimism' in keyword:
            self.txn_link = 'https://optimistic.etherscan.com/tx/{}'.format(self.txn_hash)
        elif 'arbitrum' in keyword:
            self.txn_link = 'https://arbiscan.io/tx/{}'.format(self.txn_hash)
        elif 'avalanche' in keyword:
            self.txn_link = 'https://snowtrace.io/tx/{}'.format(self.txn_hash)
        elif 'fantom' in keyword:
            self.txn_link = 'https://ftmscan.com/tx/{}'.format(self.txn_hash)
        elif 'gnosis' in keyword:
            self.txn_link = 'https://gnosisscan.io/tx/{}'.format(self.txn_hash)
        elif 'base' in keyword:
            self.txn_link = 'https://basescan.org/tx/{}'.format(self.txn_hash)
    
    def _process_statement(self, statement, is_for_statement=False):
        pattern = r"title=\'([^()]+)\s+\((0x[a-fA-F0-9]{40})\)\'>"
        matches = re.findall(pattern, statement)
        name = ''
        address = ''
        if matches:
            name, address = matches[0]
        else:
            pattern = r"title=\'(0x[a-fA-F0-9]{40})\'\>"
            matches = re.findall(pattern, statement)
            if matches:
                address = matches[0]
            else:
                pattern = r"title=\'[^(]*\((0x[a-fA-F0-9]{40})\)\'\>"
                matches = re.findall(pattern, statement)
                if matches:
                    address = matches[0]
        if is_for_statement:
            if 'title=\'' in name:
                name = name.split('title=')[-1].strip().strip("'")
            value = pattern = r"\$([\d,]+\.\d+)</a>"
            value_found = re.findall(value, statement)
            if value_found:
                value = value_found[0]
            else:
                value = -1
            count = statement.split('</span>')[0].split('>')[-1].strip()
            return name, address, count, value
        else:
            return name, address
        
    def _find_address_info(self, content):
        from_address = ''
        to_address = []
        if '</i>From:' in content:
            from_address_statement = content.split('</i>From:')[1].split('</span>')[0]
            pattern = r"0x[a-fA-F0-9]{40}"
            matches = set(re.findall(pattern, from_address_statement.lower()))
            if len(matches) == 1:
                from_address = list(matches)[0]
        else:
            pass
        if 'Interacted With (To):' in content:
            to_address_statement = content.split('Interacted With (To):')[1].split('</i></div> ')[0]
            pattern = r"0x[a-fA-F0-9]{40}"
            matches = set(re.findall(pattern, to_address_statement.lower()))
            to_address = list(matches)
        elif 'To:' in content:
            to_address_statement = content.split('To:')[1].split('</span>')[0]
            pattern = r"0x[a-fA-F0-9]{40}"
            matches = set(re.findall(pattern, to_address_statement.lower()))
            to_address = list(matches)
        else:
            pass
        return {
            'from_address': from_address,
            'to_address': to_address
        }

    
    def _get_token_transferred(self, content):
        pattern = r'([A-Za-z0-9\-]+)\s+Tokens\s+Transferred:'
        token_list = re.findall(pattern, content)
        graph = nx.DiGraph()
        for token in token_list:
            string_to_be_identified = f'{token} Tokens Transferred:'
            temp = content.split(string_to_be_identified)[1].split('From</span>')
            for t in temp:
                if 'To</span>' in t:
                    ttemp = t.split('To</span>')
                    from_statement = ttemp[0]
                    if 'For</span>' in ttemp[1]:
                        ttemp = ttemp[1].split('For</span>')
                        to_statement = ttemp[0]
                        for_statement = ttemp[1]
                        from_name, from_address = self._process_statement(from_statement)
                        to_name, to_address = self._process_statement(to_statement)
                        for_name, for_address, count, value = self._process_statement(for_statement, is_for_statement=True)
                        graph.add_node(from_address, label=f'{from_address} [{from_name}]')
                        graph.add_node(to_address, label=f'{to_address} [{to_name}]')
                        graph.add_edge(from_address, to_address, label=f'{for_address} [{for_name}] ({count})')
        return nx.node_link_data(graph,edges='edges')

    # temporary function to be removed
    def revise_token_transferred(self):
        os.makedirs(f'{self.cache_path}_revised', exist_ok=True)
        for file in os.listdir(self.cache_path):
            if file.endswith('.json'):
                data = json.load(open(os.path.join(self.cache_path, file), 'r'))
                if 'token_transfer' in data:
                    token_transfer = data['token_transfer']
                    for node in token_transfer['nodes']:
                        node['label'] = node['label'].split('[')[0].strip() if '[' in node['label'] else node['label']
                    for edge in token_transfer['edges']:
                        if len(edge['label']) > 100:
                            matches = re.findall(r'\(([0-9,]+(?:\.[0-9]+)?)\)', edge['label'])
                            amount = matches[0] if matches else ''
                            if 'title=' in edge['label']:
                                label = edge['label'].split('title=')[-1].strip().split(']')[0].strip().strip("'")
                                address = edge['label'].split('[')[0].strip()
                                edge['label'] = f'{address} [{label}] ({amount})'
                    data['token_transfer'] = token_transfer
                with open(os.path.join(f'{self.cache_path}_revised', file), 'w') as f:
                    json.dump(data, f, indent=2)
    
    def run(self, txn_hash: str, chain: str) -> dict:
        """
        Collect transaction information for given transaction hash and chain.

        Args:
            txn_hash: Transaction hash
            chain: Chain identifier (eth, bsc, poly, etc.)

        Returns:
            Dictionary containing transaction information
        """
        # Build transaction link and fetch content
        self.txn_hash = txn_hash
        self.get_transaction_link(chain)
        print("txn link:", self.txn_link)
        content = self._fetch_website_content(self.txn_link)

        if content:
            print("content 抓取成功")
        else :
            print("content 抓取失败")
            return {}


        # Parse transaction info
        status_info = self._get_transaction_status(content)
        if status_info == 'Success':
            print("transaction status is '{}'".format(status_info))
        else :
            print("transaction status is 失败")
            return {}

        block_info = self._get_transaction_block(content)
        tx_date = self._find_and_convert_datetimes(content)
        address_info = self._find_address_info(content)
        token_transfer = self._get_token_transferred(content)
        gas = self._get_transaction_gas(content)

        output = {
            "status": status_info,
            "gas_consumption": gas,
            "block_number": block_info,
            "address_info": address_info,
            "transaction_hash": txn_hash,
            "transaction_date": tx_date,
            "token_transfer": token_transfer
        }
       # print(output)
        return output
      
                       
if __name__ == "__main__":
    # df = pandas.read_csv('./transaction_links.csv')
    # df = df.dropna(subset=['txn_link'])
    # df['txn_link'] = df['txn_link'].apply(lambda x: eval(x))
    # for txn_links in tqdm(df['txn_link'].to_list()[868:]):
    #     for txn_link in txn_links:
    #         TransactionInfoCollector(txn_link).run()
    # TransactionInfoCollector('').revise_token_transferred()
    
    # df = pandas.read_csv('./dumps/transaction_links_defi.csv')
    # df = df.dropna(subset=['attack_tx'])
    # df['attack_tx'] = df['attack_tx'].apply(eval)
    # txn_links_list = df['attack_tx'].tolist()
    # collector = TransactionInfoCollector()
    # for txn_links in tqdm(txn_links_list):
    #     for txn_link in txn_links:
    #         TransactionInfoCollector().run_with_txn_link(txn_link)
            
    # collector = TransactionInfoCollector()
    # df = pandas.read_csv('./dumps/transaction_links_dappfl.csv')
    # for _,row in tqdm(df.iterrows()):
    #     txn_hash = row['transaction_hash']
    #     chain = row['chain']
    #     collector.run_with_txn_hash(txn_hash, chain)
    
    
    collector = TransactionInfoCollector()
    df = pandas.read_csv('./dumps/transaction_links_metatrustalert.csv')
    for _,row in tqdm(df.iterrows()):
        collector.run_with_txn_link(row['attack_tx'])


        