module L1DCache (
  input wire clk,
  input wire rst,
  input wire [31:0] address,
  output reg [31:0] data,
  output reg valid
);

  // Memory Request and Response Signals
  reg [31:0] memory_request_address;
  reg memory_request_valid;
  wire memory_response_ready;
  wire [31:0] memory_response_data;
  reg [31:0] dataOriginal;

  // Compressed data signals
  wire [7:0] compressed_data;
  wire [7:0] decompressed_data;

  // Define cache parameters
  parameter CACHE_SIZE = 2048; // 2 KB
  parameter CACHE_WAY = 4;
  parameter CACHE_LINE_SIZE = 32; // 32 bytes



  // Declare cache storage
  reg [31:0] cache_data [0:$clog2(CACHE_SIZE / CACHE_LINE_SIZE) - 1][0:CACHE_WAY - 1];
  reg [31:0] cache_tags [0:$clog2(CACHE_SIZE / CACHE_LINE_SIZE) - 1][0:CACHE_WAY - 1];
  reg cache_valid [0:$clog2(CACHE_SIZE / CACHE_LINE_SIZE )- 1][0:CACHE_WAY - 1];

  // Cache state
  reg [31:0] cache_out_data;
  reg cache_hit;

  // Compression logic
  AxDeduplicationCompressor compressor (
    .uncompressed_data_stream(cache_out_data),
    .compressed_data(compressed_data)
  );

  // Counter for data transfer visualization
  reg [2:0] transfer_counter;

  // Implement cache logic 
  always @* begin
    // Cache hit logic
    cache_hit = 0;
    for (integer way = 0; way < CACHE_WAY; way = way + 1) begin
      if (cache_valid[address[CACHE_SIZE-1:CACHE_LINE_SIZE]][way] &&
          cache_tags[address[CACHE_SIZE-1:CACHE_LINE_SIZE]][way] == address[CACHE_SIZE-1:5]) begin
        cache_hit = 1;
        // Set cache_out_data to the data from the cache
        cache_out_data = cache_data[address[CACHE_SIZE-1:CACHE_LINE_SIZE]][way];
      end
    end
  end

  // Logic for handling cache misses and initiating main memory request
  always @(posedge clk or posedge rst) begin
    if (rst) begin
      // Reset cache and other signals
      for (integer i = 0; i < $clog2(CACHE_SIZE / CACHE_LINE_SIZE); i = i + 1) begin
        for (integer way = 0; way < CACHE_WAY; way = way + 1) begin
          cache_valid[i][way] = 0;
        end
      end
      transfer_counter <= 0;
      // ...
    end else begin
      // Output valid signal
      valid <= ~cache_hit; // Set valid if it's a cache miss

      // Output data (4-byte word) if cache hit
      dataOriginal <= cache_out_data;
      data <= decompressed_data;

      if (~cache_hit && ~memory_request_valid) begin
        // Cache miss handling
        memory_request_address = address;
        memory_request_valid = 1;
      end

      // Wait for the memory response
      if (memory_response_ready && memory_request_valid) begin
        // Retrieve the data from the memory response
        cache_out_data = memory_response_data;

        // Update cache tags, data, and valid bit for the corresponding set
        cache_tags[address[CACHE_SIZE-1:CACHE_LINE_SIZE]][0] = address[CACHE_SIZE-1:5];
        cache_data[address[CACHE_SIZE-1:CACHE_LINE_SIZE]][0] = memory_response_data;
        cache_valid[address[CACHE_SIZE-1:CACHE_LINE_SIZE]][0] = 1;

        // Increment transfer counter for visualization
        transfer_counter <= transfer_counter + 1;

        // Memory request completed
        memory_request_valid = 0;
      end
    end
  end

  // Display data transfer sequence in the simulation
  always @(posedge clk) begin
    if (transfer_counter == 8) begin
      $display("Data Transfer Sequence: %h", cache_out_data);
      transfer_counter <= 0;
    end
  end

endmodule
