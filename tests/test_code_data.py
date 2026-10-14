import copy
import unittest

from code_data import load_dataset, parse_orders, reference_report, verify_report, render_reports


class CodeDataTests(unittest.TestCase):
    def setUp(self):
        self.rows = parse_orders('order_id,product,channel,status\n1,A,web,refunded\n2,A,app,cancelled\n3,B,web,completed\n')

    def test_denominator_includes_cancelled_and_all_groups_are_required(self):
        expected = {'groups': [{'name': 'A', 'orders': 2, 'refunded': 1, 'refund_rate': .5},
                               {'name': 'B', 'orders': 1, 'refunded': 0, 'refund_rate': 0.0}], 'total_orders': 3}
        self.assertEqual(reference_report(self.rows, 'product'), expected)
        self.assertTrue(verify_report(expected, self.rows, 'product')['passed'])
        self.assertFalse(verify_report({**expected, 'groups': expected['groups'][:1]}, self.rows, 'product')['passed'])

    def test_invalid_counts_rates_duplicates_and_extra_fields_do_not_pass(self):
        good = reference_report(self.rows, 'product')
        bad = []
        for key, value in [('orders', True), ('refunded', -1), ('refund_rate', float('nan')),
                           ('refund_rate', True), ('refund_rate', 1), ('orders', 1)]:
            changed = copy.deepcopy(good)
            changed['groups'][0][key] = value
            bad.append(changed)
        bad.extend([{**good, 'groups': good['groups'] * 2}, {**good, 'total_orders': True},
                    {**good, 'unexpected': 'x'}, None, {'groups': [], 'total_orders': 3}])
        for result in bad:
            with self.subTest(result=result):
                self.assertFalse(verify_report(result, self.rows, 'product')['passed'])

    def test_fixture_ids_and_csv_rows_are_validated(self):
        for name in ('orders-demo', 'orders-edge'):
            dataset = load_dataset(name)
            self.assertGreater(len(dataset['rows']), 0)
            self.assertEqual(len(dataset['sha256']), 64)
        with self.assertRaises(ValueError): load_dataset('../../.env')
        for text in ('order_id,product,channel,status\n1,A,web,refunded\n1,A,app,paid\n',
                     'order_id,product,channel,status\n1,A,web,unknown\n',
                     'order_id,product,channel,status\n', 'other\nanything\n'):
            with self.subTest(text=text), self.assertRaises(ValueError): parse_orders(text)

    def test_rounded_candidate_does_not_change_threshold_decision(self):
        rows = load_dataset('orders-edge')['rows']
        candidate = reference_report(rows, 'product')
        for group in candidate['groups']:
            group['refund_rate'] = round(group['refund_rate'], 6)
        verified = verify_report(candidate, rows, 'product')
        self.assertTrue(verified['passed'])
        mixed = next(group for group in verified['report']['groups'] if group['name'] == '混合商品')
        self.assertEqual(mixed['refund_rate'], 1 / 3)
        reports = render_reports(verified['report'], title='边界', group_by='product', threshold=1 / 3)
        self.assertIn('| 混合商品 | 3 | 1 | 33.33% | 是 |', reports['report.md'])
        self.assertIn('混合商品,3,1,0.333333,是', reports['report.csv'])
