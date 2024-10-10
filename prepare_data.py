from Preprocessing.converter import process_folder
from Preprocessing.CLI import parser

args = parser.parse_args()
process_folder(args.input, args.mode, args.patient_state, args.output)
