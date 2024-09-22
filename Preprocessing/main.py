from converter import process_folder
from CLI import parser

args = parser.parse_args()
process_folder(args.input, args.mode, args.patient_state, args.output)

