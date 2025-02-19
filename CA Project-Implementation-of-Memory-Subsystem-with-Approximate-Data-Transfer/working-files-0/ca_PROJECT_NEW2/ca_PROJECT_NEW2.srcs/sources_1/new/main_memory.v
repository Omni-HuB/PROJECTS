module MainMemory (
  input wire clk,
  input wire valid,
  output wire ready,
  input wire [31:0] data
);
  // Parameters for main memory size
  parameter MEM_SIZE = 8192; // 8 KB
  parameter WORD_SIZE = 4;   // 4 bytes

  // Calculate the number of locations
  localparam NUM_LOCATIONS = MEM_SIZE / WORD_SIZE;

  // State and data arrays
  reg [31:0] memory [0:NUM_LOCATIONS-1];
  reg [31:0] read_data;
  reg valid_request;

  // Logic for main memory access
  always @(posedge clk) begin
    if (valid_request) begin
      // Read data from memory (simplified)
      read_data <= memory[data[31:0]/WORD_SIZE];

      // Additional logic for write operations (not implemented in detail)
      // For now, just set the memory location with the provided data
      memory[data[31:0]/WORD_SIZE] <= data[31:0];
    end
  end

  // Output data to the L1D Cache
  assign ready = 1; // This is simplified, and you might need more complex logic
  assign data = read_data;

endmodule
