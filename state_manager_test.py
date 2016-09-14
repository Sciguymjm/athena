import enum
import math
import re
import unittest

from state import State
from state import PlayerType
from state import Character
import state_manager
from state_manager import int_handler
from state_manager import float_handler
from state_manager import generic_wrapper
from state_manager import add_address

class TestEnum(enum.Enum):
    ValOne = 1
    ValTwo = 2

class IntHandlerTest(unittest.TestCase):
    def setUp(self):
        self.state = State()

    def test_int_handler_basic(self):
        with self.assertRaises(AttributeError):
            self.state.attribute
        handler = int_handler(self.state, 'attribute')
        self.assertEqual(self.state.attribute, 0)
        handler(b'\x00\x00\x00\x01')
        self.assertEqual(self.state.attribute, 1)

    def test_int_handler_mask(self):
        handler = int_handler(self.state, 'attribute', mask=0xFF00)
        handler(b'\x00\x00\xFF\xFF')
        self.assertEqual(self.state.attribute, 0xFF00)

    def test_int_handler_shift(self):
        handler = int_handler(self.state, 'attribute', shift=8)
        handler(b'\xF0\xFF\xFF\x0F')
        self.assertEqual(self.state.attribute, 0xFFF0FFFF)

    def test_int_handler_wrapper(self):
        wrapper = lambda x: x*x
        handler = int_handler(self.state, 'attribute', wrapper=wrapper)
        handler(b'\x00\x00\x00\x04')
        self.assertEqual(self.state.attribute, 0x10)

    def test_int_handler_default(self):
        handler = int_handler(self.state, 'attribute', default=7)
        self.assertEqual(self.state.attribute, 7)
        handler(b'\x00\x00\x00\x08')
        self.assertEqual(self.state.attribute, 8)

    def test_int_handler_enum(self):
        handler = int_handler(self.state, 'attribute', wrapper=TestEnum)
        handler(b'\x00\x00\x00\x02')
        self.assertEqual(self.state.attribute, TestEnum.ValTwo)

class FloatHandlerTest(unittest.TestCase):
    def setUp(self):
        self.state = State()

    def test_float_handler_basic(self):
        with self.assertRaises(AttributeError):
            self.state.attribute
        handler = float_handler(self.state, 'attribute')
        self.assertEqual(self.state.attribute, 0.0)
        handler(b'@I\x0f\xdb')
        self.assertAlmostEqual(self.state.attribute, math.pi, places=5)

    def test_float_handler_wrapper(self):
        handler = float_handler(self.state, 'attribute', lambda x: x*x)
        handler(b'@I\x0f\xdb')
        self.assertAlmostEqual(self.state.attribute, math.pi*math.pi, places=5)

    def test_float_handler_default(self):
        handler = float_handler(self.state, 'attribute', default=7.0)
        self.assertEqual(self.state.attribute, 7.0)

    def test_float_handler_wrapper_default(self):
        handler = float_handler(self.state, 'attribute', lambda x: x > 0, True)
        self.assertTrue(self.state.attribute)
        handler(b'\xc2(\x00\x00')
        self.assertFalse(self.state.attribute)
        handler(b'B(\x00\x00')
        self.assertTrue(self.state.attribute)

class GenericWrapperTest(unittest.TestCase):
    def test_generic_wrapper_basic(self):
        self.assertEqual(generic_wrapper(1, None, 0), 1)
        self.assertEqual(generic_wrapper(1, lambda x: 2*x, 0), 2)

    def test_generic_wrapper_enum(self):
        self.assertEqual(generic_wrapper(1, TestEnum, 0), TestEnum.ValOne)
        self.assertEqual(generic_wrapper(3, TestEnum, 0), 0)

class AddAddressTest(unittest.TestCase):
    def test_add_address(self):
        self.assertEqual(add_address('0', 0), '00000000')
        self.assertEqual(add_address('0', 1), '00000001')
        self.assertEqual(add_address('1', 0), '00000001')
        self.assertEqual(add_address('1', 1), '00000002')
        self.assertEqual(add_address('804530E0', 0xE90), '80453F70')

class StateManagerTest(unittest.TestCase):
    def setUp(self):
        self.state = State()
        self.state_manager = state_manager.StateManager(self.state)

    def test_state_manager_basic(self):
        self.assertEqual(self.state.frame, 0)
        self.state_manager.handle('804D7420', b'\x00\x00\x00\x01')
        self.assertEqual(self.state.frame, 1)

    def test_state_manager_player(self):
        self.assertEqual(self.state.players[0].character, Character.Unselected)
        self.assertEqual(self.state.players[0].type, PlayerType.Unselected)
        self.state_manager.handle('803F0E08', b'\x00\x00\x0A\x00')
        self.assertEqual(self.state.players[0].character, Character.Fox)
        self.assertEqual(self.state.players[0].type, PlayerType.Human)

    def test_state_manager_asserts(self):
        with self.assertRaises(AssertionError):
            self.state_manager.handle('missing', 12345)

    def test_state_manager_locations(self):
        for location in self.state_manager.locations():
            self.assertRegex(location, '[0-9A-F]{7}[048C]( [0-9A-F]*[048C])*')

if __name__ == '__main__':
    unittest.main()
