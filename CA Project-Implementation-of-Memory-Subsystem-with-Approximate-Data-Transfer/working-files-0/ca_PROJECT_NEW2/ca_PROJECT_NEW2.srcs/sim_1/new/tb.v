module MemorySubsystemTop_tb;

  // Parameters
  localparam CLK_PERIOD = 10; // Clock period in time units

  // Signals
  reg clk;
  reg [31:0] l1d_address;
  reg l1d_valid;
  wire l1d_ready;
  wire [31:0] l1d_data;

  // Instantiate the MemorySubsystemTop module
  MemorySubsystemTop dut (
    .clk(clk),
    .l1d_address(l1d_address),
    .l1d_valid(l1d_valid),
    .l1d_ready(l1d_ready),
    .l1d_data(l1d_data)
  );

// Clock generation
always begin
  #(CLK_PERIOD / 2) clk = ~clk;
end
  // Test stimulus
  initial begin
    // Initialize signals
    clk = 0;
    l1d_address = 32'h00000000;
    l1d_valid = 0;

    // Apply reset
    #5 l1d_valid = 1; // Assume a rising edge triggered reset
    #5 l1d_valid = 0;

    // Test sequence
    #5 l1d_address = 32'h00001234; // Assuming a cache miss
    #5 l1d_valid = 1;

    // Wait for l1d_ready to be asserted
    #10 while (!l1d_ready) #1;

    // The main memory should have received the address, and l1d_data should contain the read data
    $display("Read data from Main Memory: %h", l1d_data);

    // Add more test sequences as needed

    // End simulation
    #10 $finish;
  end

endmodule
